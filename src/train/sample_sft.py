"""Phase 6 helper: sample the SFT model on the DPO prompts (D39). Run as a subprocess so vLLM's
GPU memory is released before scoring and DPO.

  python -m src.train.sample_sft <prompts.jsonl> <sft_adapter_dir> <out.jsonl> <base_model> <time_budget_s> [limit]

vLLM runs on both T4s (tensor parallel: ~3x the KV cache of one T4, D38). Prompts go in
groups of BATCH, in file order, SAMPLES answers each; no new group starts after time_budget_s
(counted from launch), so the step cannot run into the 3 h Kaggle cap. Writes one row per
sampled prompt: {episode_id, outputs: [text, ...]}; src.train.score_samples scores them.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

from src.data.chunk import read_jsonl, write_jsonl
from src.infer.generator import VLLMGenerator

SAMPLES = 2
BATCH = 16  # prompts per vLLM call; vLLM keeps as many of the 32 sequences running as the KV cache holds


def main(argv: list[str]) -> int:
    t0 = time.time()
    prompts_path, adapter, out_path, base_model, budget_s = argv[:5]
    limit = int(argv[5]) if len(argv) > 5 else None
    prompts = list(read_jsonl(Path(prompts_path)))[:limit]
    gen = VLLMGenerator(base_model, tensor_parallel_size=2, max_lora_rank=16, max_model_len=8192,
                        gpu_memory_utilization=0.9, temperature=0.9)
    gen.use_adapter("sft", adapter)
    rows = []
    for start in range(0, len(prompts), BATCH):
        if time.time() - t0 > float(budget_s):
            break
        batch = prompts[start:start + BATCH]
        texts = gen.chat([p["prompt"] for p in batch for _ in range(SAMPLES)], max_tokens=2200)
        rows += [{"episode_id": p["episode_id"], "outputs": texts[i * SAMPLES:(i + 1) * SAMPLES]}
                 for i, p in enumerate(batch)]
        write_jsonl(Path(out_path), rows)  # saved after every group
    print(f"[sample_sft] {len(rows)}/{len(prompts)} prompts sampled x{SAMPLES} in {time.time() - t0:.0f}s "
          f"({gen.generated_tokens / max(gen.generate_seconds, 1):.0f} tok/s)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
