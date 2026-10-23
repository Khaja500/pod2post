"""Phase 4/7 evaluation (kernel 05_eval): run every system on the held-out test set and score it.

Generation: vLLM split across both T4s (tensor parallel, D38), base model in fp16, LoRA
adapters switched per system; times are reported per episode and as T4-seconds (wall x GPUs)
for the cost estimate. Scoring: NLI on GPU 1 in the memory vLLM leaves free, then the metrics
in src.eval.run_eval. Writes outputs/<system>.jsonl, scores/<system>.jsonl and results.json.
A system listed in `notes_from` skips Stage A and writes its articles from another system's
notes (SFT + DPO from SFT's: DPO trained on Stage B only, D38).

Resumable (the full eval is ~13 GPU-hours, D37): episodes go through the pipeline
`episodes_per_batch` at a time, each batch is appended to outputs/<system>.jsonl and its
generation stats to outputs/<system>.stats.json, and no new batch starts after `time_budget_s`.
A system is scored as soon as all its episodes are generated. A rerun with --resume copies
the previous run's outputs/ and scores/ forward and continues; a system whose adapter was
retrained since (adapter_run, D39) is regenerated, and so is a system that writes from its
notes. results.json lists the scored systems, the pending ones, and stopped_by_time_budget.
"""
from __future__ import annotations

import json
import shutil
import time
from collections import defaultdict
from pathlib import Path

from src.data.chunk import read_jsonl, write_jsonl
from src.data.testset import HASH_FILE, id_hash, load_hashes
from src.eval.run_eval import aggregate, score_episode
from src.infer import pipeline


def test_episodes(data_dir: Path, smoke: bool) -> tuple[list[dict], dict[str, list[str]]]:
    titles = {e["episode_id"]: e["title"] for e in read_jsonl(data_dir / "episodes.jsonl") if e["split"] == "test"}
    if {id_hash(e) for e in titles} != load_hashes(data_dir / HASH_FILE):
        raise RuntimeError("test episodes in the dataset do not match the frozen test_ids.sha256")
    chunks = defaultdict(list)
    for c in read_jsonl(data_dir / "chunks.jsonl"):
        if c["split"] == "test":
            chunks[c["episode_id"]].append(c["text"])
    ids = sorted(titles, key=lambda e: len(chunks[e]))[:2] if smoke else sorted(titles)
    return [{"episode_id": e, "title": titles[e], "chunks": chunks[e]} for e in ids], chunks


def _load(path: Path) -> list[dict]:
    return list(read_jsonl(path)) if path.is_file() else []


def _append(path: Path, rows: list[dict]) -> None:
    with path.open("a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def gen_stats(stats: dict, gpus: int, notes_from: str | None) -> dict:
    """Generation stats summed over every batch and run of one system."""
    return {"tokens_per_s": round(stats["generated_tokens"] / max(stats["generate_seconds"], 1e-6), 1),
            "seconds_per_episode": round(stats["generation_wall_s"] / max(stats["episodes"], 1), 1),
            "generated_tokens": stats["generated_tokens"], "generation_wall_s": round(stats["generation_wall_s"]),
            "gpus": gpus, "notes_from": notes_from, "adapter_run": stats.get("adapter_run")}


def _started_at(manifest: Path) -> str | None:
    return json.loads(manifest.read_text()).get("started_at") if manifest.is_file() else None


def adapter_run(adapter: str | None) -> str | None:
    """Start time of the training run that wrote the adapter (its kernel's run_manifest.json)."""
    return _started_at(Path(adapter).parent / "run_manifest.json") if adapter else None


def stale_systems(out_dir: Path, systems: dict[str, str | None], notes_from: dict, resume_dir: Path | None) -> set[str]:
    """Systems whose carried-over outputs came from another adapter (it was retrained since), plus
    the systems that write from their notes. Stats without `adapter_run` predate the field: they
    are stale when the adapter's run started after the run they were resumed from."""
    resumed_from = _started_at(resume_dir / "run_manifest.json") if resume_dir is not None else None
    stale = set()
    for name, adapter in systems.items():
        path, current = out_dir / "outputs" / f"{name}.stats.json", adapter_run(adapter)
        if current is None or not path.is_file():
            continue
        made_with = json.loads(path.read_text()).get("adapter_run")
        if made_with:
            changed = made_with != current
        else:
            changed = resumed_from is not None and current > resumed_from
        if changed:
            stale.add(name)
    return stale | {name for name, source in notes_from.items() if source in stale and name in systems}


def run(data_dir: Path, out_dir: Path, cfg: dict, systems: dict[str, str | None], smoke: bool,
        resume_dir: Path | None = None, gen=None, scorer=None) -> dict:
    """gen / scorer are injected by tests; on Kaggle they are built on first use."""
    t0 = time.time()
    budget = cfg["smoke_time_budget_s"] if smoke else cfg["time_budget_s"]
    episodes, chunks = test_episodes(data_dir, smoke)
    for sub in ("outputs", "scores"):
        (out_dir / sub).mkdir(parents=True, exist_ok=True)
        if resume_dir is not None and (resume_dir / sub).is_dir():
            for f in (resume_dir / sub).iterdir():
                if f.is_file() and not (out_dir / sub / f.name).is_file():
                    shutil.copy(f, out_dir / sub / f.name)
    for name in sorted(stale_systems(out_dir, systems, cfg.get("notes_from", {}), resume_dir)):
        print(f"[eval] {name}: adapter retrained since its outputs were made; regenerating", flush=True)
        for f in (f"outputs/{name}.jsonl", f"outputs/{name}.stats.json", f"scores/{name}.jsonl"):
            (out_dir / f).unlink(missing_ok=True)
    g, size, gpus = cfg["generation"], cfg["episodes_per_batch"], cfg["generation"].get("tensor_parallel_size", 1)
    summary = {"episodes": len(episodes), "smoke": smoke, "systems": {}, "pending": [], "stopped_by_time_budget": False}
    for name, adapter in systems.items():
        out_path, stats_path = out_dir / "outputs" / f"{name}.jsonl", out_dir / "outputs" / f"{name}.stats.json"
        done = {r["episode_id"] for r in _load(out_path)}
        todo = [e for e in episodes if e["episode_id"] not in done]
        stats = json.loads(stats_path.read_text()) if stats_path.is_file() else \
            {"episodes": 0, "generated_tokens": 0, "generate_seconds": 0.0, "generation_wall_s": 0.0}
        stats.setdefault("adapter_run", adapter_run(adapter))
        source = cfg.get("notes_from", {}).get(name)
        source = source if source in systems else None
        prior = {r["episode_id"]: r for r in _load(out_dir / "outputs" / f"{source}.jsonl")} if source else {}
        for start in range(0, len(todo), size):
            if time.time() - t0 > budget:
                summary["stopped_by_time_budget"] = True
                break
            if gen is None:
                from src.infer.generator import VLLMGenerator

                gen = VLLMGenerator(cfg["base_model"], tensor_parallel_size=gpus,
                                    max_lora_rank=16 if any(systems.values()) else 0,
                                    max_model_len=g["max_model_len"], gpu_memory_utilization=g["gpu_memory_utilization"],
                                    temperature=g["temperature"], top_p=g["top_p"], top_k=g["top_k"])
            batch = todo[start:start + size]
            gen.use_adapter(name if adapter else None, adapter)
            gen.generated_tokens, gen.generate_seconds = 0, 0.0
            tb = time.time()
            reuse = prior if source and all(e["episode_id"] in prior for e in batch) else None
            _append(out_path, pipeline.run(batch, gen, g["max_tokens"], g["notes_budget_tokens"], notes_from=reuse))
            stats = {**stats, "episodes": stats["episodes"] + len(batch),
                     "generated_tokens": stats["generated_tokens"] + gen.generated_tokens,
                     "generate_seconds": stats["generate_seconds"] + gen.generate_seconds,
                     "generation_wall_s": stats["generation_wall_s"] + time.time() - tb}
            stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
            print(f"[eval] {name}: {len(done) + start + len(batch)}/{len(episodes)} episodes, "
                  f"{time.time() - t0:.0f}s", flush=True)
        results = _load(out_path)
        if len(results) < len(episodes):
            summary["pending"].append(name)
            continue
        score_path = out_dir / "scores" / f"{name}.jsonl"
        rows = _load(score_path)
        if len(rows) < len(episodes):
            if scorer is None:
                import torch

                from src.eval.faithfulness import NLIScorer

                device = "cuda:1" if torch.cuda.device_count() > 1 else "cuda:0"
                scorer = NLIScorer(cfg["faithfulness"]["nli_model"], device=device,
                                   batch_size=cfg["faithfulness"].get("batch_size", 32))
            rows = [score_episode(r, chunks[r["episode_id"]], scorer) for r in results]
            write_jsonl(score_path, rows)
        summary["systems"][name] = aggregate(rows, gen_stats(stats, gpus, source))
        print(f"[eval] {name}: {json.dumps(summary['systems'][name])}", flush=True)
    (out_dir / "results.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
