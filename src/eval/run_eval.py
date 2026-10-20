"""Score one system's pipeline outputs on the held-out test set (spec §6).

Per episode: faithfulness (NLI), hallucinated entities/numbers vs the transcript, SEO
checklist, Flesch reading ease, social-pack format. A missing or unparseable article
scores 0 on faithfulness and SEO, so failures are penalised rather than skipped.
"""
from __future__ import annotations

import statistics

from src.data.filter import checkable_text, grounded_entity, ner_entities, norm, ungrounded_numbers
from src.eval.faithfulness import faithfulness
from src.eval.seo import article_parts, seo_checks, social_checks

FLESCH_TARGET = (50, 70)


def flesch(article: str) -> float:
    import textstat

    return round(textstat.flesch_reading_ease(article_parts(article)["body"]), 1)


def score_episode(result: dict, chunks: list[str], scorer) -> dict:
    article, social = result.get("article"), result.get("social")
    row = {"episode_id": result["episode_id"], "article_present": article is not None,
           "stage_a_invalid": result.get("stage_a_invalid", 0), "n_chunks": result.get("n_chunks"),
           "social": social_checks(social)}
    if article is None:
        return {**row, "faithfulness": 0.0, "seo_score": 0.0, "hallucinated_entities": None,
                "hallucinated_numbers": None, "flesch": None, "words": 0}
    transcript = "\n".join(chunks)
    src = norm(transcript)
    generated = checkable_text(article, social)
    seo = seo_checks(article, result["primary_keyword"])
    faith = faithfulness(article, chunks, scorer)
    bad_entities = [e for e in ner_entities(generated) if not grounded_entity(e, src)]
    bad_numbers = ungrounded_numbers(generated, transcript)
    return {**row, "faithfulness": faith["score"], "faithfulness_detail": faith, "seo_score": seo["score"],
            "seo": seo, "hallucinated_entities": len(bad_entities), "hallucinated_entity_examples": bad_entities[:5],
            "hallucinated_numbers": len(bad_numbers), "hallucinated_number_examples": bad_numbers[:5],
            "flesch": flesch(article), "words": seo["words"]}


def aggregate(rows: list[dict], gen_stats: dict) -> dict:
    present = [r for r in rows if r["article_present"]]

    def mean(key, pool=rows):
        vals = [r[key] for r in pool if r.get(key) is not None]
        return round(statistics.mean(vals), 4) if vals else None

    checks = present[0]["seo"]["checks"].keys() if present else []
    return {
        "episodes": len(rows),
        "article_present_rate": round(len(present) / max(len(rows), 1), 4),
        "faithfulness": mean("faithfulness"),
        "hallucinated_entities_per_article": mean("hallucinated_entities", present),
        "hallucinated_numbers_per_article": mean("hallucinated_numbers", present),
        "articles_with_any_hallucination": round(sum(
            (r["hallucinated_entities"] or 0) + (r["hallucinated_numbers"] or 0) > 0 for r in present)
            / max(len(present), 1), 4),
        "seo_score": mean("seo_score"),
        "seo_check_pass_rates": {c: round(sum(r["seo"]["checks"][c] for r in present) / len(present), 4)
                                 for c in checks},
        "flesch": mean("flesch", present),
        "flesch_in_target_rate": round(sum(FLESCH_TARGET[0] <= (r["flesch"] or -1) <= FLESCH_TARGET[1]
                                           for r in present) / max(len(present), 1), 4),
        "social_valid_json_rate": round(sum(r["social"]["valid_json"] for r in rows) / max(len(rows), 1), 4),
        "x_posts_ok_rate": round(sum(r["social"]["x_posts_ok"] for r in rows) / max(len(rows), 1), 4),
        "stage_a_invalid_rate": round(sum(r["stage_a_invalid"] for r in rows) /
                                      max(sum(r["n_chunks"] or 0 for r in rows), 1), 4),
        "words": mean("words", present),
        **gen_stats,
    }
