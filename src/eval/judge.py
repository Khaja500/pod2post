"""LLM judge (spec §6): each system's article vs the base model's, judged by the teacher model.

Every pair is judged twice with the A/B order swapped, to cancel position bias. The judge sees
transcript passages retrieved for both articles (BM25 over transcript windows), so it can
check claims rather than judge style alone. Per episode a system wins if it wins one order
and does not lose the other; it loses symmetrically; anything else is a tie.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from src.data.chunk import read_jsonl
from src.eval.faithfulness import BM25, article_sentences, transcript_windows

JUDGE_SYSTEM = "You are a strict editor comparing two articles written from the same podcast transcript."
JUDGE_USER = """Transcript passages (the source of truth):
<<<
{transcript}
>>>

Article A:
<<<
{a}
>>>

Article B:
<<<
{b}
>>>

Compare the articles on: faithfulness to the transcript (no invented facts, names, numbers or quotes), \
usefulness and clarity for a reader, and SEO structure (title, meta description, headings, FAQ). \
Faithfulness matters most. Reply with one line "Winner: A", "Winner: B" or "Winner: tie", then one \
sentence with the main reason."""
VERDICT_RE = re.compile(r"winner\s*:\s*\**\s*(a|b|tie)\b", re.I)


def evidence(chunks: list[str], articles: list[str], count, budget: int = 3500) -> str:
    """Transcript windows that best match the articles' sentences, in transcript order."""
    windows = transcript_windows(chunks)
    if not windows:
        return ""
    bm25 = BM25(windows)
    picked = []
    for art in articles:
        for s in article_sentences(art):
            picked += bm25.top(s, 1)
    keep, used = [], 0
    for i in sorted(set(picked), key=picked.index):
        n = count(windows[i])
        if used + n > budget:
            break
        keep.append(i)
        used += n
    return "\n...\n".join(windows[i] for i in sorted(keep))


def verdict(text: str) -> str | None:
    m = VERDICT_RE.search(text or "")
    return m[1].lower() if m else None


def combine(sys_first: str | None, base_first: str | None) -> str:
    """sys_first: verdict with the system as A; base_first: verdict with the system as B."""
    votes = [{"a": 1, "b": -1}.get(sys_first, 0), {"b": 1, "a": -1}.get(base_first, 0)]
    if max(votes) == 1 and min(votes) >= 0:
        return "win"
    if min(votes) == -1 and max(votes) <= 0:
        return "loss"
    return "tie"


def run(eval_dir: Path, chunks: dict[str, list[str]], gen, out_dir: Path, max_tokens: int = 80) -> dict:
    outputs = {p.stem: {r["episode_id"]: r for r in read_jsonl(p)} for p in (eval_dir / "outputs").glob("*.jsonl")}
    base = outputs.pop("base")
    jobs, convs = [], []
    for name, rows in sorted(outputs.items()):
        for ep, r in sorted(rows.items()):
            a, b = r.get("article"), base.get(ep, {}).get("article")
            if not (a and b):
                continue
            ev = evidence(chunks[ep], [a, b], gen.count)
            for order, (x, y) in (("sys_first", (a, b)), ("base_first", (b, a))):
                jobs.append((name, ep, order))
                convs.append([{"role": "system", "content": JUDGE_SYSTEM},
                              {"role": "user", "content": JUDGE_USER.format(transcript=ev, a=x, b=y)}])
    texts = gen.chat(convs, max_tokens, temperature=0.0)
    votes: dict = {}
    for (name, ep, order), t in zip(jobs, texts):
        votes.setdefault(name, {}).setdefault(ep, {})[order] = (verdict(t), t.strip()[:300])
    summary, rows = {}, []
    for name, eps in votes.items():
        results = []
        for ep, v in eps.items():
            outcome = combine(v.get("sys_first", (None,))[0], v.get("base_first", (None,))[0])
            results.append(outcome)
            rows.append({"system": name, "episode_id": ep, "outcome": outcome,
                         **{k: {"verdict": x[0], "text": x[1]} for k, x in v.items()}})
        n = len(results)
        summary[name] = {k: round(results.count(k) / n, 4) for k in ("win", "tie", "loss")} | {"episodes": n}
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "judge.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with (out_dir / "judge_details.jsonl").open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return summary
