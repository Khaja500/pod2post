"""Write reports/results.md from pulled 05_eval output folders (spec §6).

  .venv/bin/python -m src.eval.report runs/<date>/05_eval-v<N> [runs/... more] [--judge judge.json] [--out results.md]

Later folders override earlier ones per system, so the baseline run (Phase 4) and the
final run (Phase 7) can be combined. Every number in the report comes from results.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from src.data.chunk import read_jsonl

REPO = Path(__file__).resolve().parents[2]
ORDER = ["base", "sft", "sft_dpo", "teacher"]
LABELS = {"base": "Base (zero-shot)", "sft": "SFT", "sft_dpo": "SFT + DPO", "teacher": "Teacher (upper bound)"}


def pct(x):
    return "—" if x is None else f"{100 * x:.1f}%"


def num(x, digits=2):
    return "—" if x is None else f"{x:.{digits}f}"


def load(run_dirs: list[Path]) -> tuple[dict, dict, dict, dict]:
    systems, outputs, scores, sources = {}, {}, {}, {}
    for d in run_dirs:
        res = json.loads((d / "results.json").read_text())
        for name, agg in res["systems"].items():
            systems[name] = agg
            outputs[name] = {r["episode_id"]: r for r in read_jsonl(d / "outputs" / f"{name}.jsonl")}
            scores[name] = {r["episode_id"]: r for r in read_jsonl(d / "scores" / f"{name}.jsonl")}
            sources[name] = d.name
    return systems, outputs, scores, sources


def headline(systems: dict) -> str | None:
    best = "sft_dpo" if "sft_dpo" in systems else "sft" if "sft" in systems else None
    if not best or "base" not in systems:
        return None
    b, t = systems["base"], systems[best]
    rel = (t["faithfulness"] - b["faithfulness"]) / max(b["faithfulness"], 1e-9)
    halluc = b["hallucinated_entities_per_article"], t["hallucinated_entities_per_article"]
    cut = (halluc[0] - halluc[1]) / halluc[0] if halluc[0] else None
    return (f"A QLoRA-fine-tuned 4B model ({LABELS[best]}), trained on Kaggle's free T4 GPUs, writes articles that are "
            f"**{100 * rel:+.0f}%** more faithful to the transcript than the base model "
            f"({pct(b['faithfulness'])} → {pct(t['faithfulness'])} of sentences supported), "
            + (f"with **{100 * cut:.0f}% fewer** invented entities per article, " if cut is not None else "")
            + f"and an SEO checklist score of **{pct(t['seo_score'])}** (base {pct(b['seo_score'])}), "
            f"at {num(t['seconds_per_episode'], 0)} s per episode on one T4.")


def main(argv: list[str]) -> int:
    out, judge_path = REPO / "reports" / "results.md", None
    for flag in ("--out", "--judge"):
        if flag in argv:
            i = argv.index(flag)
            value, argv = Path(argv[i + 1]), argv[:i] + argv[i + 2:]
            out, judge_path = (value, judge_path) if flag == "--out" else (out, value)
    if not argv:
        print(__doc__)
        return 1
    systems, outputs, scores, sources = load([Path(a) for a in argv])
    names = [n for n in ORDER if n in systems]
    n_eps = max(s["episodes"] for s in systems.values())
    lines = ["# Results", "",
             f"Held-out test set: {n_eps} episodes (38 TalkRL + 12 Changelog), frozen before any data generation. "
             "Every system runs the same map-reduce pipeline (chunk notes → merged notes → article + social pack) "
             "with the same prompts and sampling. Numbers come from `results.json` in the runs listed at the bottom.", ""]
    if (h := headline(systems)):
        lines += ["**Headline:** " + h, ""]
    lines += ["## Main comparison", "",
              "| System | Faithfulness ↑ | Invented entities / article ↓ | Invented numbers / article ↓ | "
              "Articles with any invention ↓ | SEO score ↑ | Flesch (target 50–70) | Valid social JSON ↑ | "
              "X posts ≤ 280 ↑ | Tokens/s (1×T4) | s / episode |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
    for n in names:
        s = systems[n]
        lines.append(f"| {LABELS[n]} | {pct(s['faithfulness'])} | {num(s['hallucinated_entities_per_article'])} | "
                     f"{num(s['hallucinated_numbers_per_article'])} | {pct(s['articles_with_any_hallucination'])} | "
                     f"{pct(s['seo_score'])} | {num(s['flesch'], 1)} ({pct(s['flesch_in_target_rate'])} in range) | "
                     f"{pct(s['social_valid_json_rate'])} | {pct(s['x_posts_ok_rate'])} | {num(s['tokens_per_s'], 0)} | "
                     f"{num(s['seconds_per_episode'], 0)} |")
    checks = list(next(iter(systems.values()))["seo_check_pass_rates"])
    lines += ["", "## SEO checklist pass rates", "", "| Check | " + " | ".join(LABELS[n] for n in names) + " |",
              "|---|" + "---|" * len(names)]
    for c in checks:
        lines.append(f"| `{c}` | " + " | ".join(pct(systems[n]["seo_check_pass_rates"].get(c)) for n in names) + " |")
    lines += ["", "Other: article produced " + ", ".join(f"{LABELS[n]} {pct(systems[n]['article_present_rate'])}" for n in names)
              + "; invalid Stage A JSON " + ", ".join(f"{LABELS[n]} {pct(systems[n]['stage_a_invalid_rate'])}" for n in names) + "."]
    if judge_path and judge_path.is_file():
        judge = json.loads(judge_path.read_text())
        lines += ["", "## LLM judge (pairwise vs base, teacher model as judge)", "",
                  "| System | Win | Tie | Loss |", "|---|---|---|---|"]
        lines += [f"| {LABELS[k]} | {pct(v['win'])} | {pct(v['tie'])} | {pct(v['loss'])} |" for k, v in judge.items()]
        lines += ["", "Each pair is judged in both A/B orders; a system wins an episode only if it wins one order and "
                  "does not lose the other."]

    lines += ["", "## Side-by-side examples", ""]
    common = sorted(set.intersection(*(set(outputs[n]) for n in names)))
    picks = [e for e in common if e.startswith("talkrl:")][:2] + [e for e in common if e.startswith("changelog:")][:1]
    for ep in picks:
        lines += [f"### {outputs[names[0]][ep]['title']} (`{ep}`)", ""]
        for n in names:
            r, sc = outputs[n][ep], scores[n][ep]
            art = (r.get("article") or "(no article produced)").split()
            lines += [f"**{LABELS[n]}** · keyword *{r['primary_keyword']}* · faithfulness {pct(sc['faithfulness'])} · "
                      f"SEO {pct(sc['seo_score'])} · invented entities {sc.get('hallucinated_entities')}", "",
                      "> " + " ".join(art[:220]) + (" …" if len(art) > 220 else ""), ""]
    lines += ["## Runs", ""] + [f"- {LABELS[n]}: `{sources[n]}`" for n in names]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
