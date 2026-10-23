"""Phase 6 helper: score the SFT samples and the teacher's answer for every sampled DPO prompt
with the evaluation's own scorer (D39), so DPO can prefer the more faithful answer.

  python -m src.train.score_samples <samples.jsonl> <prompts.jsonl> <chunks.jsonl> <out.jsonl> <nli_model> [batch_size]

Runs after sample_sft has exited, so the NLI model has a whole T4. Per candidate: faithfulness
(NLI), invented entities/numbers, SEO score, Flesch, words (src.eval.run_eval.score_episode,
strict parsing like the eval) plus the training-data checks (src.data.filter.stage_b_problems).
"""
from __future__ import annotations

import sys
import time
from collections import defaultdict
from pathlib import Path

from src.data.chunk import read_jsonl, write_jsonl
from src.data.filter import stage_b_problems
from src.eval.faithfulness import NLIScorer
from src.eval.run_eval import score_episode
from src.eval.seo import parse_output

KEEP = ("faithfulness", "seo_score", "words", "flesch", "hallucinated_entities", "hallucinated_numbers")


def score_candidate(text: str, prompt: dict, chunks: list[str], scorer) -> dict:
    article, social = parse_output(text)
    row = score_episode({"episode_id": prompt["episode_id"], "article": article, "social": social,
                         "primary_keyword": prompt["primary_keyword"]}, chunks, scorer)
    problems = stage_b_problems(text, prompt["notes_text"], "\n".join(chunks), "")
    if not row["social"]["valid_json"]:
        problems.append("social_not_strict_json")
    return {"text": text, "problems": problems, **{k: row[k] for k in KEEP}}


def main(argv: list[str]) -> int:
    t0 = time.time()
    samples_path, prompts_path, chunks_path, out_path, nli_model = argv[:5]
    batch_size = int(argv[5]) if len(argv) > 5 else 32
    prompts = {p["episode_id"]: p for p in read_jsonl(Path(prompts_path))}
    samples = list(read_jsonl(Path(samples_path)))
    wanted = {s["episode_id"] for s in samples}
    chunks = defaultdict(list)
    for c in read_jsonl(Path(chunks_path)):
        if c["episode_id"] in wanted:
            chunks[c["episode_id"]].append(c["text"])
    scorer = NLIScorer(nli_model, device="cuda:0", batch_size=batch_size)
    rows = []
    for s in samples:
        p, ch = prompts[s["episode_id"]], chunks[s["episode_id"]]
        rows.append({"episode_id": s["episode_id"],
                     "teacher": score_candidate(p["chosen"], p, ch, scorer),
                     "sft": [score_candidate(t, p, ch, scorer) for t in s["outputs"]]})
    write_jsonl(Path(out_path), rows)
    print(f"[score_samples] {len(rows)} prompts scored in {time.time() - t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
