"""Phase 4/7 evaluation (kernel 05_eval): run every system on the held-out test set and score it.

Generation: vLLM on GPU 0 only (the spec's cost metric is "on a T4"), base model in fp16,
LoRA adapters switched per system. Scoring: NLI on the second T4, then the metrics in
src.eval.run_eval. Writes outputs/<system>.jsonl, scores/<system>.jsonl and results.json.

Resumable (the full eval is ~13 GPU-hours, D37): episodes go through the pipeline
`episodes_per_batch` at a time, each batch is appended to outputs/<system>.jsonl and its
generation stats to outputs/<system>.stats.json, and no new batch starts after `time_budget_s`.
A system is scored as soon as all its episodes are generated. A rerun with --resume copies
the previous run's outputs/ and scores/ forward and continues. results.json lists the scored
systems, the pending ones, and stopped_by_time_budget.
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


def gen_stats(stats: dict) -> dict:
    """Generation stats summed over every batch and run of one system."""
    return {"tokens_per_s": round(stats["generated_tokens"] / max(stats["generate_seconds"], 1e-6), 1),
            "seconds_per_episode": round(stats["generation_wall_s"] / max(stats["episodes"], 1), 1),
            "generated_tokens": stats["generated_tokens"], "generation_wall_s": round(stats["generation_wall_s"])}


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
    g, size = cfg["generation"], cfg["episodes_per_batch"]
    summary = {"episodes": len(episodes), "smoke": smoke, "systems": {}, "pending": [], "stopped_by_time_budget": False}
    for name, adapter in systems.items():
        out_path, stats_path = out_dir / "outputs" / f"{name}.jsonl", out_dir / "outputs" / f"{name}.stats.json"
        done = {r["episode_id"] for r in _load(out_path)}
        todo = [e for e in episodes if e["episode_id"] not in done]
        stats = json.loads(stats_path.read_text()) if stats_path.is_file() else \
            {"episodes": 0, "generated_tokens": 0, "generate_seconds": 0.0, "generation_wall_s": 0.0}
        for start in range(0, len(todo), size):
            if time.time() - t0 > budget:
                summary["stopped_by_time_budget"] = True
                break
            if gen is None:
                from src.infer.generator import VLLMGenerator

                gen = VLLMGenerator(cfg["base_model"], max_lora_rank=16 if any(systems.values()) else 0,
                                    max_model_len=g["max_model_len"], gpu_memory_utilization=g["gpu_memory_utilization"],
                                    temperature=g["temperature"], top_p=g["top_p"], top_k=g["top_k"])
            batch = todo[start:start + size]
            gen.use_adapter(name if adapter else None, adapter)
            gen.generated_tokens, gen.generate_seconds = 0, 0.0
            tb = time.time()
            _append(out_path, pipeline.run(batch, gen, g["max_tokens"], g["notes_budget_tokens"]))
            stats = {"episodes": stats["episodes"] + len(batch),
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

                device = "cuda:1" if torch.cuda.device_count() > 1 else "cuda:0"  # vLLM holds GPU 0
                scorer = NLIScorer(cfg["faithfulness"]["nli_model"], device=device)
            rows = [score_episode(r, chunks[r["episode_id"]], scorer) for r in results]
            write_jsonl(score_path, rows)
        summary["systems"][name] = aggregate(rows, gen_stats(stats))
        print(f"[eval] {name}: {json.dumps(summary['systems'][name])}", flush=True)
    (out_dir / "results.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
