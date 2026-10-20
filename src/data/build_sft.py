"""Phase 3: filtered teacher outputs -> chat-format SFT data (one adapter, two tasks).

  .venv/bin/python -m src.data.build_sft runs/<date>/02_teacher_gen-v<N>/teacher

Writes data/processed/sft.jsonl: {id, task: "A"|"B", episode_id, split: "train"|"val", messages}.
Targets are canonical: Stage A notes re-serialised as JSON, Stage B rebuilt as
<article>...</article><social>{json}</social> from the parsed teacher output. 5% of episodes go to "val" (whole episodes, so no leakage).
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import yaml

from src.data.chunk import hf_token_counter, read_jsonl, write_jsonl
from src.data.prompts import stage_a_messages, stage_b_messages
from src.data.testset import HASH_FILE, assert_no_test_leak
from src.eval.seo import parse_output

REPO = Path(__file__).resolve().parents[2]
VAL_SHARE = 0.05
MAX_SEQ = 4096          # spec §2 training length
TEMPLATE_OVERHEAD = 8   # chat-template tokens per message


def canonical_output(article: str, social: dict) -> str:
    return f"<article>\n{article}\n</article>\n<social>\n{json.dumps(social, ensure_ascii=False)}\n</social>"


def main(argv: list[str]) -> int:
    if len(argv) != 1:
        print(__doc__)
        return 1
    teacher_dir = Path(argv[0])
    data_dir = REPO / "data" / "processed"
    cfg = yaml.safe_load((REPO / "configs" / "teacher.yaml").read_text())
    rng = random.Random(cfg["seed"])
    chunks = {(c["episode_id"], c["chunk_idx"]): c for c in read_jsonl(data_dir / "chunks.jsonl")}
    titles = {e["episode_id"]: e["title"] for e in read_jsonl(data_dir / "episodes.jsonl")}

    a_rows = list(read_jsonl(teacher_dir / "filtered_stage_a.jsonl"))
    b_rows = list(read_jsonl(teacher_dir / "filtered_stage_b.jsonl"))
    a_rows = rng.sample(a_rows, min(len(a_rows), cfg["targets"]["stage_a"]))
    b_rows = b_rows[: cfg["targets"]["stage_b"]]
    episodes = sorted({r["episode_id"] for r in a_rows + b_rows})
    assert_no_test_leak(episodes, data_dir / HASH_FILE)
    val_eps = set(rng.sample(episodes, max(1, round(len(episodes) * VAL_SHARE))))

    rows = []
    for r in a_rows:
        c = chunks[(r["episode_id"], r["chunk_idx"])]
        msgs = stage_a_messages(titles[r["episode_id"]], c["text"], c["chunk_idx"] + 1, c["n_chunks"])
        msgs.append({"role": "assistant", "content": json.dumps(r["notes"], ensure_ascii=False)})
        rows.append({"id": f"A:{r['episode_id']}:{r['chunk_idx']}", "task": "A", "episode_id": r["episode_id"],
                     "split": "val" if r["episode_id"] in val_eps else "train", "messages": msgs})
    for r in b_rows:
        msgs = stage_b_messages(r["title"], r["notes_text"], r["primary_keyword"], r["secondary_keywords"])
        article, social = parse_output(r["output"])
        msgs.append({"role": "assistant", "content": canonical_output(article, social)})
        rows.append({"id": f"B:{r['episode_id']}", "task": "B", "episode_id": r["episode_id"],
                     "primary_keyword": r["primary_keyword"],
                     "split": "val" if r["episode_id"] in val_eps else "train", "messages": msgs})
    count = hf_token_counter(yaml.safe_load((REPO / "configs" / "data.yaml").read_text())["tokenizer"])
    for r in rows:
        r["n_tokens"] = sum(count(m["content"]) + TEMPLATE_OVERHEAD for m in r["messages"])
    too_long = [r for r in rows if r["n_tokens"] > MAX_SEQ]
    rows = [r for r in rows if r["n_tokens"] <= MAX_SEQ]
    print(f"dropped {len(too_long)} examples longer than {MAX_SEQ} tokens "
          f"({sum(r['task'] == 'B' for r in too_long)} Stage B)")
    rng.shuffle(rows)
    n = write_jsonl(data_dir / "sft.jsonl", rows)
    counts = {f"{t}/{s}": sum(r["task"] == t and r["split"] == s for r in rows) for t in "AB" for s in ("train", "val")}
    print(f"wrote {n} SFT examples -> {data_dir / 'sft.jsonl'}: {counts}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
