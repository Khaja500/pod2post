"""Stage B output parsing plus the SEO and format rule checks (spec §4.3 and §6).

Used both to filter teacher data and to score every system in evaluation, so the
rules the student is trained on and the rules it is graded on are the same code.
"""
from __future__ import annotations

import json
import re

ARTICLE_RE = re.compile(r"<article>\s*(.*?)\s*</article>", re.S)
SOCIAL_RE = re.compile(r"<social>\s*(.*?)\s*(?:</social>|$)", re.S)
JSON_OBJ_RE = re.compile(r"\{.*\}", re.S)
META_RE = re.compile(r"^\**meta description:?\**:?\s*(.+)$", re.I)
FAQ_RE = re.compile(r"^##\s+(faq|frequently asked questions)\b", re.I)
WORD_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9'’-]*")

TITLE_MAX, META_MAX, X_POST_MAX = 60, 160, 280
MIN_H2, DENSITY = 3, (0.005, 0.025)
SOCIAL_KEYS = ["linkedin", "x_thread", "newsletter", "pull_quotes"]


EARLY_CLOSE_RE = re.compile(r'"\s*\}\s*,?\s*\n\s*"(?=\w+"\s*:)')
TRAILING_COMMA_RE = re.compile(r",\s*([\]}])")


def _load_json(text: str, repair: bool):
    try:
        return json.loads(text)
    except ValueError:
        if not repair:
            return None
    try:  # teacher slips: object closed after the first key, trailing commas
        return json.loads(TRAILING_COMMA_RE.sub(r"\1", EARLY_CLOSE_RE.sub('",\n"', text)))
    except ValueError:
        return None


def parse_output(text: str, repair: bool = False) -> tuple[str | None, dict | None]:
    """-> (article markdown, social dict); None for a part that is missing or invalid.

    repair=True fixes two common JSON slips; it is used only for teacher data, whose
    targets are re-serialised anyway. Evaluation parses strictly.
    """
    a = ARTICLE_RE.search(text or "")
    article = a[1].strip() if a else None
    social = None
    s = SOCIAL_RE.search(text or "")
    if s and (m := JSON_OBJ_RE.search(s[1])):
        social = _load_json(m[0], repair)
    return article, social if isinstance(social, dict) else None


def article_parts(article: str) -> dict:
    title, meta, h2, h3, body, faq_q, in_faq = None, None, [], [], [], 0, False
    for line in article.splitlines():
        s = line.strip()
        if s.startswith("# ") and title is None:
            title = s[2:].strip()
        elif META_RE.match(s) and meta is None:
            meta = META_RE.match(s)[1].strip()
        elif s.startswith("## "):
            h2.append(s[3:].strip())
            in_faq = bool(FAQ_RE.match(s))
            body.append(s[3:])
        elif s.startswith("### "):
            h3.append(s[4:].strip())
            faq_q += in_faq
            body.append(s[4:])
        else:
            body.append(s)
    return {"title": title, "meta": meta, "h2": h2, "h3": h3, "faq_questions": faq_q,
            "faq": any(FAQ_RE.match(f"## {h}") for h in h2), "body": "\n".join(body).strip()}


def words(text: str) -> list[str]:
    return WORD_RE.findall(text)


def keyword_count(text: str, keyword: str) -> int:
    kw = keyword.strip()
    if not kw:
        return 0
    pattern = r"(?<![A-Za-z0-9])" + r"\s+".join(map(re.escape, kw.split())) + r"(?![A-Za-z0-9])"
    return len(re.findall(pattern, text, re.I))


def seo_checks(article: str, keyword: str) -> dict:
    """The 8-item checklist from spec §6; `score` is the fraction passed."""
    p = article_parts(article)
    body_words = words(p["body"])
    density = keyword_count(p["body"], keyword) / max(len(body_words), 1)
    checks = {
        "kw_in_title": bool(p["title"]) and keyword_count(p["title"], keyword) > 0,
        "kw_in_first_100_words": keyword_count(" ".join(body_words[:100]), keyword) > 0,
        "kw_in_h2": any(keyword_count(h, keyword) for h in p["h2"]),
        "title_len_ok": bool(p["title"]) and len(p["title"]) <= TITLE_MAX,
        "meta_len_ok": bool(p["meta"]) and len(p["meta"]) <= META_MAX,
        "min_3_h2": len(p["h2"]) >= MIN_H2,
        "faq_present": p["faq"] and p["faq_questions"] >= 1,
        "density_ok": DENSITY[0] <= density <= DENSITY[1],
    }
    return {"checks": checks, "score": sum(checks.values()) / len(checks), "density": round(density, 4),
            "words": len(body_words), "title_chars": len(p["title"] or ""), "meta_chars": len(p["meta"] or "")}


def social_checks(social: dict | None) -> dict:
    ok_json = isinstance(social, dict) and all(k in social for k in SOCIAL_KEYS)
    thread = social.get("x_thread") if ok_json else None
    posts_ok = isinstance(thread, list) and bool(thread) and all(
        isinstance(t, str) and 0 < len(t) <= X_POST_MAX for t in thread)
    quotes = social.get("pull_quotes") if ok_json else None
    return {"valid_json": ok_json, "x_posts_ok": posts_ok,
            "x_posts": len(thread) if isinstance(thread, list) else 0,
            "pull_quotes": len(quotes) if isinstance(quotes, list) else 0}


def format_problems(article: str | None, social: dict | None) -> list[str]:
    """Hard format rules for filtering (spec §4.3). Empty list = passes."""
    if article is None:
        return ["missing_article"]
    p = article_parts(article)
    problems = []
    if not p["title"]:
        problems.append("missing_title")
    elif len(p["title"]) > TITLE_MAX:
        problems.append("title_too_long")
    if not p["meta"]:
        problems.append("missing_meta")
    elif len(p["meta"]) > META_MAX:
        problems.append("meta_too_long")
    if len(p["h2"]) < MIN_H2:
        problems.append("too_few_h2")
    if not p["faq"]:
        problems.append("missing_faq")
    s = social_checks(social)
    if not s["valid_json"]:
        problems.append("invalid_social_json")
    elif not s["x_posts_ok"]:
        problems.append("x_post_too_long")
    return problems
