"""Phase 4/7 evaluation (kernel 05_eval): run every system on the held-out test set and score it.

Generation: vLLM on GPU 0 only (the spec's cost metric is "on a T4"), base model in fp16,
LoRA adapters switched per system. Scoring: NLI on the second T4, then the metrics in
src.eval.run_eval. Writes outputs/<system>.jsonl, scores/<system>.jsonl and results.json.
"""
from __future__ import annotations

import json
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


def run(data_dir: Path, out_dir: Path, cfg: dict, systems: dict[str, str | None], smoke: bool) -> dict:
    from src.infer.generator import VLLMGenerator

    episodes, chunks = test_episodes(data_dir, smoke)
    (out_dir / "outputs").mkdir(parents=True, exist_ok=True)
    (out_dir / "scores").mkdir(parents=True, exist_ok=True)
    g = cfg["generation"]
    gen = VLLMGenerator(cfg["base_model"], max_lora_rank=16 if any(systems.values()) else 0,
                        max_model_len=g["max_model_len"], gpu_memory_utilization=g["gpu_memory_utilization"],
                        temperature=g["temperature"], top_p=g["top_p"], top_k=g["top_k"])
    gen_stats = {}
    for name, adapter in systems.items():
        gen.use_adapter(name if adapter else None, adapter)
        gen.generated_tokens, gen.generate_seconds = 0, 0.0
        t0 = time.time()
        results = pipeline.run(episodes, gen, g["max_tokens"], g["notes_budget_tokens"])
        wall = time.time() - t0
        write_jsonl(out_dir / "outputs" / f"{name}.jsonl", results)
        gen_stats[name] = {"tokens_per_s": round(gen.generated_tokens / max(gen.generate_seconds, 1e-6), 1),
                           "seconds_per_episode": round(wall / len(episodes), 1),
                           "generated_tokens": gen.generated_tokens, "generation_wall_s": round(wall)}
        print(f"[eval] {name}: {len(episodes)} episodes in {wall:.0f}s, {gen_stats[name]['tokens_per_s']} tok/s",
              flush=True)

    import torch

    from src.eval.faithfulness import NLIScorer

    device = "cuda:1" if torch.cuda.device_count() > 1 else "cuda:0"  # vLLM still holds GPU 0
    scorer = NLIScorer(cfg["faithfulness"]["nli_model"], device=device)
    summary = {"episodes": len(episodes), "smoke": smoke, "systems": {}}
    for name in systems:
        results = list(read_jsonl(out_dir / "outputs" / f"{name}.jsonl"))
        rows = [score_episode(r, chunks[r["episode_id"]], scorer) for r in results]
        write_jsonl(out_dir / "scores" / f"{name}.jsonl", rows)
        summary["systems"][name] = aggregate(rows, gen_stats[name])
        print(f"[eval] {name}: {json.dumps(summary['systems'][name])}", flush=True)
    (out_dir / "results.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
