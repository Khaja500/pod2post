"""Phase 6: DPO preference pairs (spec §4.4). Four kinds of pair:
  (a) on-policy (D39): the SFT model's least faithful sample (sampled and scored on Kaggle in
      04_dpo) vs. the most faithful clean answer among the teacher's and the other samples
  (b) teacher output vs. the same with an injected hallucination: a fake statistic + an invented claim
  (c) teacher output vs. a keyword-stuffed version of it
  (d) teacher output vs. a generic/vague rewrite of it (made by the teacher in 02_teacher_gen)
(a) wherever the scores give one; (b)-(d) round-robin over the other prompts as a third of the total.

  .venv/bin/python -m src.data.build_dpo runs/<date>/02_teacher_gen-v<N>/teacher [out_dir]
writes data/processed/dpo_synthetic.jsonl with (b), (c), (d) candidates for every prompt.
The 04_dpo kernel adds (a) and calls assemble() to pick the final mix.
"""
from __future__ import annotations

import json
import random
import re
import sys
from pathlib import Path

import yaml

from src.data.build_sft import canonical_output
from src.data.chunk import read_jsonl, write_jsonl
from src.data.prompts import stage_b_messages
from src.data.testset import HASH_FILE, assert_no_test_leak
from src.eval.seo import DENSITY, format_problems, keyword_count, parse_output, words

REPO = Path(__file__).resolve().parents[2]
N_PROMPTS = 300
TYPES = ["sft_less_faithful", "hallucination", "keyword_stuffed", "generic"]

FAKE_ORGS = ["Northbridge Analytics", "the Helios Institute", "Meridian Research Group", "Cobalt Insights",
             "the Larkspur Foundation", "Vantage Metrics"]
STAT_TEMPLATES = [
    "According to a {year} survey by {org}, {pct}% of listeners say {kw} has changed how they work.",
    "Recent data from {org} suggests that {kw} adoption grew by {pct}% between {year} and {year2}.",
    "A {year} report from {org} found that {n:,} organisations now invest in {kw}.",
]
CLAIM_TEMPLATES = [
    "{speaker} also revealed that {org} will partner with the show's guests on a new {kw} project in {year2}.",
    "{speaker} added that their team cut costs by {pct}% after rethinking {kw}.",
]
SPEAKERS_RE = re.compile(r"^Speakers: (.+)$", re.M)


def _fake(rng: random.Random, kw: str, speaker: str) -> tuple[str, str]:
    year = rng.randint(2019, 2024)
    vals = dict(kw=kw, org=rng.choice(FAKE_ORGS), pct=rng.randint(17, 89), year=year, year2=year + rng.randint(1, 3),
                n=rng.randint(12, 950) * 100, speaker=speaker)
    return rng.choice(STAT_TEMPLATES).format(**vals), rng.choice(CLAIM_TEMPLATES).format(**vals)


def _body_paragraphs(lines: list[str]) -> list[int]:
    return [i for i, line in enumerate(lines) if len(line.split()) >= 25 and not line.lstrip().startswith(("#", "Meta"))]


def inject_hallucination(article: str, social: dict, kw: str, notes_text: str, rng: random.Random):
    """(b): one fake statistic + one invented speaker claim in the article, and one in the thread."""
    m = SPEAKERS_RE.search(notes_text)
    speaker = rng.choice(m[1].split(", ")) if m else "The host"
    stat, claim = _fake(rng, kw, speaker)
    lines = article.splitlines()
    paras = _body_paragraphs(lines)
    if len(paras) < 2:
        return None
    for i, sentence in zip(sorted(rng.sample(paras, 2)), (stat, claim)):
        lines[i] = f"{lines[i]} {sentence}"
    social = json.loads(json.dumps(social))
    if isinstance(social.get("x_thread"), list) and social["x_thread"]:
        social["x_thread"][0] = stat[:280]
    return "\n".join(lines), social


def keyword_stuff(article: str, social: dict, kw: str, rng: random.Random):
    """(c): push keyword density well past the 2.5% ceiling."""
    lines = article.splitlines()
    paras = _body_paragraphs(lines)
    if not paras:
        return None
    for i in paras:
        lines[i] = f"When it comes to {kw}, {lines[i][0].lower()}{lines[i][1:]} That is why {kw} matters."
    lines.insert(paras[0] + 1, f"\nLooking for {kw}? This {kw} guide covers {kw} basics, {kw} tips and the best "
                               f"{kw} ideas, because {kw} is what {kw} readers search for.")
    stuffed = "\n".join(lines)
    while keyword_count(stuffed, kw) / max(len(words(stuffed)), 1) <= DENSITY[1] * 1.5:
        stuffed += f"\n\nIn short: {kw}, {kw}, {kw}."
    social = json.loads(json.dumps(social))
    social["linkedin"] = f"{kw.title()}! " * 3 + str(social.get("linkedin", ""))
    return stuffed, social


def sampling_order(prompts: list[dict], seed: int, n: int = N_PROMPTS) -> list[dict]:
    """The order 04_dpo samples the SFT model in (it stops at a time budget) and assemble() uses."""
    rng = random.Random(seed)
    prompts = sorted(prompts, key=lambda p: p["episode_id"])[:n]
    rng.shuffle(prompts)
    return prompts


def faithfulness_pair(scored: dict, margin: float) -> tuple[dict, dict] | None:
    """(a), D39: rejected = the least faithful SFT sample; chosen = the most faithful clean answer
    (no check fails) among the teacher's and the other samples, kept only if it is at least
    `margin` more faithful, scores no lower on SEO and is not much shorter (DPO's length bias)."""
    sft = sorted(scored["sft"], key=lambda c: c["faithfulness"])
    if not sft:
        return None
    rejected = sft[0]
    clean = [c for c in [scored["teacher"], *sft[1:]] if not c["problems"]]
    if not clean:
        return None
    chosen = max(clean, key=lambda c: c["faithfulness"])
    if (chosen["faithfulness"] - rejected["faithfulness"] >= margin and chosen["seo_score"] >= rejected["seo_score"]
            and chosen["words"] >= 0.85 * rejected["words"]):
        return chosen, rejected
    return None


def assemble(prompts: list[dict], scored: dict[str, dict], seed: int, n: int = N_PROMPTS,
             margin: float = 0.08, synthetic_share: float = 0.33, min_pairs: int = 0) -> list[dict]:
    """A faithfulness pair for every scored prompt that has one; then synthetic pairs (b)-(d),
    round-robin over the other prompts, as `synthetic_share` of the total (or up to min_pairs)."""
    def pair(p, kind, chosen, rejected):
        return {"episode_id": p["episode_id"], "rejected_type": kind, "prompt": p["prompt"],
                "chosen": [{"role": "assistant", "content": chosen}],
                "rejected": [{"role": "assistant", "content": rejected}]}

    pairs, rest = [], []
    for p in sampling_order(prompts, seed, n):
        found = faithfulness_pair(scored[p["episode_id"]], margin) if p["episode_id"] in scored else None
        if found:
            chosen, rejected = found
            pairs.append({**pair(p, TYPES[0], chosen["text"], rejected["text"]),
                          "faithfulness": [chosen["faithfulness"], rejected["faithfulness"]]})
        else:
            rest.append(p)
    n_synthetic = round(len(pairs) * synthetic_share / (1 - synthetic_share)) if pairs else len(rest)
    n_synthetic = max(n_synthetic, min_pairs - len(pairs))
    for k, p in enumerate(rest[:n_synthetic]):
        order = TYPES[1:][k % 3:] + TYPES[1:][:k % 3]
        kind = next((t for t in order if t in p["rejected"]), None)
        if kind:
            pairs.append(pair(p, kind, p["chosen"], p["rejected"][kind]))
    return pairs


def main(argv: list[str]) -> int:
    if len(argv) not in (1, 2):
        print(__doc__)
        return 1
    teacher_dir = Path(argv[0])
    data_dir = REPO / "data" / "processed"
    out_dir = Path(argv[1]) if len(argv) == 2 else data_dir
    rng = random.Random(yaml.safe_load((REPO / "configs" / "dpo.yaml").read_text())["seed"])
    chosen_rows = list(read_jsonl(teacher_dir / "filtered_stage_b.jsonl"))
    generic = {r["episode_id"]: r["output"] for r in read_jsonl(teacher_dir / "generic.jsonl")}
    assert_no_test_leak({r["episode_id"] for r in chosen_rows}, data_dir / HASH_FILE)
    out, kinds = [], {t: 0 for t in TYPES[1:]}
    for r in chosen_rows:
        article, social = parse_output(r["output"], repair=True)
        kw = r["primary_keyword"]
        rejected = {}
        if (h := inject_hallucination(article, social, kw, r["notes_text"], rng)):
            rejected["hallucination"] = canonical_output(*h)
        if (s := keyword_stuff(article, social, kw, rng)):
            rejected["keyword_stuffed"] = canonical_output(*s)
        if r["episode_id"] in generic:
            g_article, g_social = parse_output(generic[r["episode_id"]], repair=True)
            if not format_problems(g_article, g_social):
                rejected["generic"] = canonical_output(g_article, g_social)
        for k in rejected:
            kinds[k] += 1
        out.append({"episode_id": r["episode_id"], "primary_keyword": kw, "notes_text": r["notes_text"],
                    "prompt": stage_b_messages(r["title"], r["notes_text"], kw, r["secondary_keywords"]),
                    "chosen": canonical_output(article, social), "rejected": rejected})
    n = write_jsonl(out_dir / "dpo_synthetic.jsonl", out)
    print(f"wrote {n} DPO prompts -> {out_dir / 'dpo_synthetic.jsonl'}; rejected candidates: {kinds}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
