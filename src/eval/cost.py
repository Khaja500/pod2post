"""Free cost estimate per article (configs/costs.yaml): no API is called.

Token usage per episode comes from the evaluated pipeline itself: prompt tokens are rebuilt
exactly from the test transcripts and each system's merged notes (same prompts the model saw),
completion tokens are what the model generated on Kaggle. API cost = tokens x published prices;
self-hosted cost = measured T4-seconds per episode (wall time x T4s used) x one T4's hourly price.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import yaml

from src.data.chunk import read_jsonl
from src.data.prompts import stage_a_messages, stage_b_messages

TEMPLATE_OVERHEAD = 8  # chat-template tokens per message


def prompt_tokens(result: dict, chunks: list[str], count) -> int:
    convs = [stage_a_messages(result["title"], c, i + 1, len(chunks)) for i, c in enumerate(chunks)]
    convs.append(stage_b_messages(result["title"], result["notes_text"], result["primary_keyword"],
                                  result["secondary_keywords"]))
    return sum(count(m["content"]) + TEMPLATE_OVERHEAD for conv in convs for m in conv)


def estimate(eval_dir: Path, system: str, agg: dict, chunks: dict[str, list[str]], count, cfg: dict,
             gpu_hours_used: float | None = None) -> dict:
    results = list(read_jsonl(eval_dir / "outputs" / f"{system}.jsonl"))
    p_tok = sum(prompt_tokens(r, chunks[r["episode_id"]], count) for r in results) / len(results)
    c_tok = agg["generated_tokens"] / len(results)
    api = {name: round((p_tok * p["input"] + c_tok * p["output"]) / 1e6, 5) for name, p in cfg["api_models"].items()}
    t4_seconds = agg["seconds_per_episode"] * agg.get("gpus", 1)
    hosted = {name: round(t4_seconds / 3600 * usd, 5) for name, usd in cfg["self_hosted_usd_per_hour"].items()}
    ref = hosted[cfg["reference_gpu"]]
    out = {"system": system, "prices_checked": str(cfg["checked"]), "comparator": cfg["comparator"],
           "prompt_tokens_per_episode": round(p_tok), "completion_tokens_per_episode": round(c_tok),
           "seconds_per_episode_1xT4": round(t4_seconds, 1), "api_usd_per_article": api,
           "self_hosted_usd_per_article": hosted,
           "times_cheaper_than_comparator": round(api[cfg["comparator"]] / ref, 1) if ref else None}
    if gpu_hours_used is not None:
        out["one_time_gpu_hours"] = round(gpu_hours_used, 2)
        out["one_time_usd_at_reference_gpu"] = round(gpu_hours_used * cfg["self_hosted_usd_per_hour"][cfg["reference_gpu"]], 2)
    return out


def load_inputs(repo: Path) -> tuple[dict, dict[str, list[str]]] | None:
    """(costs config, test chunks by episode) or None when either is missing."""
    cfg_path, chunks_path = repo / "configs" / "costs.yaml", repo / "data" / "processed" / "chunks.jsonl"
    if not (cfg_path.is_file() and chunks_path.is_file()):
        return None
    chunks = defaultdict(list)
    for c in read_jsonl(chunks_path):
        if c["split"] == "test":
            chunks[c["episode_id"]].append(c["text"])
    return yaml.safe_load(cfg_path.read_text()), chunks
