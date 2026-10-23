"""Transcript -> notes -> article + social pack (spec §3), for evaluation and the demo.

Map: Stage A on every chunk, batched across episodes, with one low-temperature retry
for invalid JSON. Reduce: merge each episode's notes (<= 1,800 tokens, D23), pick its
keywords, then Stage B. Works with any generator that has
chat(conversations, max_tokens, temperature=None) and count(text).
With notes_from (earlier results by episode id), Stage A is skipped and those results'
merged notes and keywords feed Stage B (05_eval's SFT + DPO reuses SFT's notes, D38).
"""
from __future__ import annotations

import time

from src.data.notes import merge_notes, parse_notes, pick_keywords, sanitize_notes
from src.data.prompts import stage_a_messages, stage_b_messages
from src.eval.seo import parse_output

NOTE_FIELDS = ("episode_id", "title", "n_chunks", "stage_a_invalid", "primary_keyword", "secondary_keywords",
               "notes_text")


def run(episodes: list[dict], gen, max_tokens: dict, notes_budget: int = 1800,
        keywords: dict | None = None, notes_from: dict[str, dict] | None = None) -> list[dict]:
    """episodes: [{episode_id, title, chunks: [text, ...]}] -> one result dict per episode."""
    t0 = time.time()
    if notes_from is not None:
        results = [{k: notes_from[e["episode_id"]][k] for k in NOTE_FIELDS} for e in episodes]
    else:
        results = map_notes(episodes, gen, max_tokens, notes_budget, keywords)
    t_map = time.time() - t0
    b_convs = [stage_b_messages(r["title"], r["notes_text"], r["primary_keyword"], r["secondary_keywords"])
               for r in results]
    t1 = time.time()
    for r, out in zip(results, gen.chat(b_convs, max_tokens["stage_b"])):
        article, social = parse_output(out)
        r.update(output=out, article=article, social=social)
    elapsed = time.time() - t0
    for r in results:  # batched across episodes, so per-episode time is throughput-based
        r["seconds_per_episode"] = round(elapsed / len(episodes), 2)
    results[0]["timing"] = {"map_s": round(t_map, 1), "reduce_s": round(time.time() - t1, 1),
                            "total_s": round(elapsed, 1), "episodes": len(episodes)}
    return results


def map_notes(episodes: list[dict], gen, max_tokens: dict, notes_budget: int,
              keywords: dict | None = None) -> list[dict]:
    """Stage A on every chunk, then each episode's merged notes and keywords (no Stage B yet)."""
    jobs = [(e, i) for e in episodes for i in range(len(e["chunks"]))]
    convs = [stage_a_messages(e["title"], e["chunks"][i], i + 1, len(e["chunks"])) for e, i in jobs]
    texts = gen.chat(convs, max_tokens["stage_a"])
    notes = [parse_notes(t) for t in texts]
    retry = [k for k, n in enumerate(notes) if n is None]
    if retry:
        for k, t in zip(retry, gen.chat([convs[k] for k in retry], max_tokens["stage_a"], temperature=0.3)):
            notes[k] = parse_notes(t)

    by_ep: dict[str, list] = {e["episode_id"]: [] for e in episodes}
    for (e, i), n in zip(jobs, notes):  # same deterministic clean-up the teacher data had
        by_ep[e["episode_id"]].append(sanitize_notes(n, e["chunks"][i]) if n is not None else None)
    results = []
    for e in episodes:
        valid = [n for n in by_ep[e["episode_id"]] if n is not None]
        primary, secondary = (keywords or {}).get(e["episode_id"]) or pick_keywords(valid)
        primary = primary or e["title"].lower()
        notes_text = merge_notes(valid, gen.count, notes_budget) if valid else ""
        results.append({"episode_id": e["episode_id"], "title": e["title"], "n_chunks": len(e["chunks"]),
                        "stage_a_invalid": len(e["chunks"]) - len(valid), "primary_keyword": primary,
                        "secondary_keywords": secondary, "notes_text": notes_text})
    return results
