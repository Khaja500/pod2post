"""Phase 3 filtering of teacher outputs (spec §4.3), with the drop rate per rule.

  .venv/bin/python -m src.data.filter runs/<date>/02_teacher_gen-v<N>/teacher

Stage A (chunk -> notes) is dropped if: the notes are invalid JSON; a quote is not in
the chunk; an entity or number in the notes is not in the chunk; it near-duplicates
another example.
Stage B (notes -> article + social) is dropped if: a format rule fails; the output names
an entity or number found in neither the notes nor the transcript; near-duplicate.
Writes filtered_stage_a.jsonl, filtered_stage_b.jsonl and filter_report.json next to the input.
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path

from src.data.chunk import read_jsonl, write_jsonl
from src.eval.seo import format_problems, parse_output

NUM_RE = re.compile(r"(?<![\w.])\d[\d,]*(?:\.\d+)?")
ENTITY_LABELS = {"PERSON", "ORG", "GPE", "LOC", "PRODUCT", "EVENT", "WORK_OF_ART", "NORP", "FAC", "LAW"}
ENTITY_ALLOW = {"faq", "seo", "linkedin", "twitter", "x", "podcast", "episode", "newsletter"}
SMALL_NUMBER_MAX = 10
HASHTAG_RE = re.compile(r"#(\w+)")  # "3 takeaways", "5 questions": structural, often spelled out in speech


def norm(text: str) -> str:
    text = text.lower().replace("’", "'").replace("“", '"').replace("”", '"')
    text = re.sub(r"'s\b", "", text)
    return " " + " ".join(re.findall(r"[a-z0-9]+", text)) + " "


def numbers(text: str) -> set[str]:
    return {n.replace(",", "").rstrip(".") for n in NUM_RE.findall(text)}


def ungrounded_numbers(output: str, source: str) -> list[str]:
    have = numbers(source)
    return sorted(n for n in numbers(output)
                  if n not in have and not (n.isdigit() and int(n) <= SMALL_NUMBER_MAX))


def grounded_entity(entity: str, source_norm: str) -> bool:
    e = norm(entity)
    if e.strip() in ENTITY_ALLOW or e in source_norm:
        return True
    tokens = [t for t in e.split() if len(t) >= 3]
    if tokens and all(f" {t} " in source_norm for t in tokens):
        return True
    compact = e.replace(" ", "")  # "#BarackObama" vs "Barack Obama"
    return len(compact) >= 6 and compact in source_norm.replace(" ", "")


@lru_cache(maxsize=1)
def _nlp():
    import spacy

    return spacy.load("en_core_web_sm", disable=["lemmatizer"])


def ner_entities(text: str) -> list[str]:
    """Entities per line, so spaCy never joins a run of title-case words across lines."""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return sorted({e.text for doc in _nlp().pipe(lines) for e in doc.ents if e.label_ in ENTITY_LABELS})


def checkable_text(article: str, social: dict | None) -> str:
    """Body prose + social text. Headings and the meta line are skipped: they restate the body
    in Title Case, which spaCy over-reads as entities."""
    body = [line for line in article.splitlines()
            if not line.lstrip().startswith("#") and not line.lower().lstrip("*").startswith("meta description")]
    social_text = []
    for v in (social or {}).values():
        social_text += v if isinstance(v, list) else [v]
    text = "\n".join(body + [str(t) for t in social_text])
    return HASHTAG_RE.sub(lambda m: re.sub(r"(?<=[a-z])(?=[A-Z])", " ", m[1]), text)


def stage_a_problems(chunk: str, notes: dict | None) -> list[str]:
    if notes is None:
        return ["invalid_json"]
    src = norm(chunk)
    problems = []
    if any(norm(q["quote"]) not in src for q in notes["quotes"]):
        problems.append("quote_not_verbatim")
    if any(not grounded_entity(e, src) for e in notes["entities"]):
        problems.append("ungrounded_entity")
    text = " ".join([notes["summary"], *notes["key_points"], *notes["numbers"], *(c["claim"] for c in notes["claims"])])
    if ungrounded_numbers(text, chunk):
        problems.append("ungrounded_number")
    return problems


def stage_b_problems(output: str, notes_text: str, transcript: str, title: str) -> list[str]:
    """Grounding is checked against the transcript (+ title) only, like the evaluation:
    an entity the notes invented must not survive into training data."""
    article, social = parse_output(output, repair=True)
    problems = format_problems(article, social)
    if article is None:
        return problems
    generated = checkable_text(article, social)
    source = "\n".join([transcript, title])
    src = norm(source)
    if any(not grounded_entity(e, src) for e in ner_entities(generated)):
        problems.append("ungrounded_entity")
    if ungrounded_numbers(generated, source):
        problems.append("ungrounded_number")
    return problems


def near_duplicates(texts: list[str], threshold: float = 0.8) -> set[int]:
    """Indices of texts whose 5-gram MinHash matches an earlier text."""
    from datasketch import MinHash, MinHashLSH

    lsh, dups = MinHashLSH(threshold=threshold, num_perm=128), set()
    for i, text in enumerate(texts):
        words = norm(text).split()
        m = MinHash(num_perm=128)
        for j in range(max(1, len(words) - 4)):
            m.update(" ".join(words[j:j + 5]).encode())
        if lsh.query(m):
            dups.add(i)
        else:
            lsh.insert(str(i), m)
    return dups


def _report(name: str, problems: list[list[str]]) -> dict:
    first = Counter(p[0] for p in problems if p)
    every = Counter(r for p in problems for r in p)
    n = len(problems)
    kept = sum(not p for p in problems)
    return {"stage": name, "generated": n, "kept": kept, "kept_rate": round(kept / max(n, 1), 3),
            "dropped_by_first_rule": dict(first.most_common()),
            "fail_rate_per_rule": {r: round(c / max(n, 1), 3) for r, c in every.most_common()}}


def main(argv: list[str]) -> int:
    if len(argv) != 1:
        print(__doc__)
        return 1
    teacher_dir = Path(argv[0])
    data_dir = Path(__file__).resolve().parents[2] / "data" / "processed"
    chunks = {(c["episode_id"], c["chunk_idx"]): c["text"] for c in read_jsonl(data_dir / "chunks.jsonl")}
    transcripts = defaultdict(list)
    for (ep, idx), text in sorted(chunks.items()):
        transcripts[ep].append(text)

    a_rows = list(read_jsonl(teacher_dir / "stage_a.jsonl"))
    a_problems = [stage_a_problems(chunks[(r["episode_id"], r["chunk_idx"])], r["notes"]) for r in a_rows]
    for i in near_duplicates([json.dumps(r["notes"]) for r in a_rows]):
        a_problems[i].append("near_duplicate")

    b_rows = list(read_jsonl(teacher_dir / "stage_b.jsonl"))
    b_problems = [stage_b_problems(r["output"], r["notes_text"], "\n".join(transcripts[r["episode_id"]]), r["title"])
                  for r in b_rows]
    for i in near_duplicates([parse_output(r["output"], repair=True)[0] or r["output"] for r in b_rows]):
        b_problems[i].append("near_duplicate")

    write_jsonl(teacher_dir / "filtered_stage_a.jsonl", (r for r, p in zip(a_rows, a_problems) if not p))
    write_jsonl(teacher_dir / "filtered_stage_b.jsonl",
                ({**r, "problems": []} for r, p in zip(b_rows, b_problems) if not p))
    write_jsonl(teacher_dir / "rejected_stage_b.jsonl",
                ({**r, "problems": p} for r, p in zip(b_rows, b_problems) if p))
    report = {"stage_a": _report("A", a_problems), "stage_b": _report("B", b_problems)}
    (teacher_dir / "filter_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
