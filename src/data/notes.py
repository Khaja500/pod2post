"""Stage A notes: parse the model's JSON, then merge one episode's notes into Stage B input.

The merged notes are capped at 2,500 tokens (spec §3). Long episodes are trimmed by
dropping the lowest-ranked items round-robin across chunks, so every part of the
episode keeps its main points.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from typing import Callable

NOTE_KEYS = ["summary", "key_points", "claims", "quotes", "entities", "numbers", "keywords"]
JSON_OBJ_RE = re.compile(r"\{.*\}", re.S)
WORD_RE = re.compile(r"[a-z0-9']+")
NULLISH = {"", "null", "none", "n/a", "na", "unknown", "-"}


def _opt(value) -> str | None:
    text = str(value).strip() if value is not None else ""
    return None if text.lower() in NULLISH else text


def parse_notes(text: str) -> dict | None:
    """Lenient: take the outermost {...}, require every key, coerce the field types."""
    m = JSON_OBJ_RE.search(text or "")
    if not m:
        return None
    try:
        raw = json.loads(m[0])
    except ValueError:
        return None
    if not isinstance(raw, dict) or any(k not in raw for k in NOTE_KEYS):
        return None

    def strings(v):
        return [str(x).strip() for x in v if str(x).strip()] if isinstance(v, list) else []

    def records(v, field):
        out = []
        for x in v if isinstance(v, list) else []:
            if isinstance(x, dict) and str(x.get(field) or "").strip():
                out.append({"speaker": _opt(x.get("speaker")), field: str(x[field]).strip(),
                            "timestamp": _opt(x.get("timestamp"))})
        return out

    return {"summary": str(raw["summary"] or "").strip(), "key_points": strings(raw["key_points"]),
            "claims": records(raw["claims"], "claim"), "quotes": records(raw["quotes"], "quote"),
            "entities": strings(raw["entities"]), "numbers": strings(raw["numbers"]),
            "keywords": strings(raw["keywords"])}


def _norm(text: str) -> str:
    return " ".join(WORD_RE.findall(text.lower()))


def _near_dup(a: str, b: str, threshold: float = 0.8) -> bool:
    wa, wb = set(a.split()), set(b.split())
    return bool(wa and wb) and len(wa & wb) / len(wa | wb) >= threshold


def pick_keywords(notes: list[dict], n_secondary: int = 3) -> tuple[str, list[str]]:
    """Most frequent keyword across chunks is primary (ties: earliest seen)."""
    counts, first = Counter(), {}
    for i, n in enumerate(notes):
        for kw in n["keywords"]:
            k = _norm(kw)
            if k:
                counts[k] += 1
                first.setdefault(k, (i, kw.strip()))
    ranked = sorted(counts, key=lambda k: (-counts[k], first[k][0]))
    picked = []
    for k in ranked:
        if not any(_near_dup(k, p, 0.6) for p in picked):
            picked.append(k)
    surface = [first[k][1].lower() for k in picked]
    return (surface[0] if surface else ""), surface[1:1 + n_secondary]


def _ts(rec: dict) -> str:
    return f"[{rec['timestamp']}] " if rec.get("timestamp") else ""


def _who(rec: dict) -> str:
    return f"{rec['speaker']}: " if rec.get("speaker") else ""


SECTIONS = [  # (heading, render one item); order = order in the merged notes
    ("Summary", lambda chunk_notes: [chunk_notes["summary"]] if chunk_notes["summary"] else []),
    ("Key points", lambda n: n["key_points"]),
    ("Claims", lambda n: [f"{_ts(c)}{_who(c)}{c['claim']}" for c in n["claims"]]),
    ("Quotes", lambda n: [f'{_ts(q)}{_who(q)}"{q["quote"]}"' for q in n["quotes"]]),
    ("Numbers", lambda n: n["numbers"]),
]
KEEP_PER_CHUNK = {"Summary": 1, "Key points": 1}  # never trimmed below this
DROP_FIRST = {"Claims": 0, "Key points": 1, "Numbers": 2, "Quotes": 3, "Summary": 4}  # within one rank


def merge_notes(notes: list[dict], count: Callable[[str], int], budget: int = 2500) -> str:
    """Render deduplicated notes for Stage B, trimmed to `budget` tokens."""
    items = []  # (section_idx, chunk_idx, rank, text)
    seen: list[str] = []
    for ci, n in enumerate(notes):
        for si, (_, render) in enumerate(SECTIONS):
            for rank, text in enumerate(render(n)):
                key = _norm(text)
                if key and not any(_near_dup(key, s) for s in seen):
                    seen.append(key)
                    items.append((si, ci, rank, text))
    entities, seen_e = [], set()
    for n in notes:
        for e in n["entities"]:
            if _norm(e) and _norm(e) not in seen_e:
                seen_e.add(_norm(e))
                entities.append(e)
    speakers = sorted({r["speaker"] for n in notes for r in n["claims"] + n["quotes"] if r.get("speaker")})

    def render(kept) -> str:
        lines = []
        if speakers:
            lines.append("Speakers: " + ", ".join(speakers))
        for si, (heading, _) in enumerate(SECTIONS):
            body = [t for (s, _, _, t) in kept if s == si]
            if body:
                lines += [f"{heading}:", *(f"- {t}" for t in body)]
        if entities:
            lines.append("Entities: " + ", ".join(entities))
        return "\n".join(lines)

    kept = sorted(items, key=lambda x: (x[0], x[1], x[2]))
    sizes = {id(x): count(x[3]) + 2 for x in kept}
    total = count(render(kept))
    # Drop highest-rank items first, round-robin over chunks, never below KEEP_PER_CHUNK.
    droppable = sorted((x for x in kept if x[2] >= KEEP_PER_CHUNK.get(SECTIONS[x[0]][0], 0)),
                       key=lambda x: (-x[2], DROP_FIRST[SECTIONS[x[0]][0]], x[1]))
    for x in droppable:
        if total <= budget:
            break
        kept.remove(x)
        total -= sizes[id(x)]
    while count(render(kept)) > budget and len(entities) > 10:
        entities.pop()
    # Very long episodes: last resort is dropping per-chunk summaries (key points stay).
    summaries = [x for x in kept if x[0] == 0]
    while count(render(kept)) > budget and summaries:
        kept.remove(summaries.pop(len(summaries) // 2))
    return render(kept)
