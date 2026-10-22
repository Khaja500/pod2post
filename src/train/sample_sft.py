"""Phase 6 helper: sample the SFT model on the DPO prompts and keep outputs that fail the checks
(rejected type (a), spec §4.4). Run as a subprocess so vLLM's GPU memory is released before DPO.

  python -m src.train.sample_sft <prompts.jsonl> <chunks.jsonl> <sft_adapter_dir> <out.jsonl> <base_model> <time_budget_s> [limit]

A sample "fails" when it breaks a format rule or names an entity or number that is in
neither the notes nor the transcript; for each prompt the first failing sample is kept.
Prompts go to vLLM a few at a time and no new batch starts after time_budget_s (counted from
launch), so the step cannot run into the 3 h Kaggle cap.
"""
from __future__ import annotations

import sys
import time
from collections import defaultdict
from pathlib import Path

from src.data.build_sft import canonical_output
from src.data.chunk import read_jsonl, write_jsonl
from src.data.filter import stage_b_problems
from src.eval.seo import parse_output
from src.infer.generator import VLLMGenerator

SAMPLES = 2
BATCH = 4  # prompts per vLLM call: fp16 weights leave the T4's KV cache room for only ~5 long sequences


def main(argv: list[str]) -> int:
    t0 = time.time()
    dpo_path, chunks_path, adapter, out_path, base_model, budget_s = argv[:6]
    limit = int(argv[6]) if len(argv) > 6 else None
    prompts = list(read_jsonl(Path(dpo_path)))[:limit]
    transcripts = defaultdict(list)
    for c in read_jsonl(Path(chunks_path)):
        transcripts[c["episode_id"]].append(c["text"])
    gen = VLLMGenerator(base_model, max_lora_rank=16, max_model_len=8192, gpu_memory_utilization=0.9, temperature=0.9)
    gen.use_adapter("sft", adapter)
    failures, sampled = {}, 0
    for start in range(0, len(prompts), BATCH):
        if time.time() - t0 > float(budget_s):
            break
        batch = prompts[start:start + BATCH]
        texts = gen.chat([p["prompt"] for p in batch for _ in range(SAMPLES)], max_tokens=2200)
        for i, p in enumerate(batch):
            for text in texts[i * SAMPLES:(i + 1) * SAMPLES]:
                problems = stage_b_problems(text, p["notes_text"], "\n".join(transcripts[p["episode_id"]]), "")
                if problems:
                    article, social = parse_output(text, repair=True)
                    failures[p["episode_id"]] = {"episode_id": p["episode_id"], "problems": problems,
                                                 "output": canonical_output(article, social) if article and social else text}
                    break
        sampled += len(batch)
    write_jsonl(Path(out_path), failures.values())
    print(f"[sample_sft] {len(failures)}/{sampled} sampled prompts have a failing SFT sample "
          f"({sampled}/{len(prompts)} sampled in {time.time() - t0:.0f}s; "
          f"{gen.generated_tokens / max(gen.generate_seconds, 1):.0f} tok/s)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
