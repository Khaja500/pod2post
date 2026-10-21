"""Phase 6 helper: sample the SFT model on the DPO prompts and keep outputs that fail the checks
(rejected type (a), spec §4.4). Run as a subprocess so vLLM's GPU memory is released before DPO.

  python -m src.train.sample_sft <dpo_synthetic.jsonl> <chunks.jsonl> <sft_adapter_dir> <out.jsonl> <base_model> [limit]

A sample "fails" when it breaks a format rule or names an entity or number that is in
neither the notes nor the transcript; for each prompt the first failing sample is kept.
"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

from src.data.build_sft import canonical_output
from src.data.chunk import read_jsonl, write_jsonl
from src.data.filter import stage_b_problems
from src.eval.seo import parse_output
from src.infer.generator import VLLMGenerator

SAMPLES = 2


def main(argv: list[str]) -> int:
    dpo_path, chunks_path, adapter, out_path, base_model = argv[:5]
    limit = int(argv[5]) if len(argv) > 5 else None
    prompts = list(read_jsonl(Path(dpo_path)))[:limit]
    transcripts = defaultdict(list)
    for c in read_jsonl(Path(chunks_path)):
        transcripts[c["episode_id"]].append(c["text"])
    gen = VLLMGenerator(base_model, max_lora_rank=16, max_model_len=8192, gpu_memory_utilization=0.9, temperature=0.9)
    gen.use_adapter("sft", adapter)
    convs = [p["prompt"] for p in prompts for _ in range(SAMPLES)]
    texts = gen.chat(convs, max_tokens=2200)
    failures = {}
    for i, p in enumerate(prompts):
        for text in texts[i * SAMPLES:(i + 1) * SAMPLES]:
            problems = stage_b_problems(text, p["notes_text"], "\n".join(transcripts[p["episode_id"]]), "")
            if problems:
                article, social = parse_output(text, repair=True)
                failures[p["episode_id"]] = {"episode_id": p["episode_id"], "problems": problems,
                                             "output": canonical_output(article, social) if article and social else text}
                break
    write_jsonl(Path(out_path), failures.values())
    print(f"[sample_sft] {len(failures)}/{len(prompts)} prompts have a failing SFT sample "
          f"({gen.generated_tokens / max(gen.generate_seconds, 1):.0f} tok/s)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
