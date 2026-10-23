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


def best_system(systems: dict) -> str | None:
    return "sft_dpo" if "sft_dpo" in systems else "sft" if "sft" in systems else None


def timed_system(systems: dict, name: str) -> str:
    """The system whose timing covers the whole pipeline for `name` (its notes source, if it reused notes)."""
    source = systems[name].get("notes_from")
    return source if source in systems else name


def t4_seconds(s: dict) -> float:
    return s["seconds_per_episode"] * s.get("gpus", 1)


def project_gpu_hours() -> float | None:
    import csv

    path = REPO / "runs" / "quota_log.csv"
    if not path.is_file():
        return None
    with path.open(newline="") as f:
        return sum(float(r.get("kernel_runtime_hours") or r.get("wall_hours") or 0) for r in csv.DictReader(f))


def cost_section(run_dirs: list[Path], systems: dict, sources: dict) -> tuple[list[str], dict | None]:
    """'Cost per article' table from configs/costs.yaml; nothing if the inputs are missing."""
    from src.eval import cost

    inputs = cost.load_inputs(REPO)
    best = best_system(systems) or ("base" if "base" in systems else None)
    name = timed_system(systems, best) if best else None
    if inputs is None or name is None or "generated_tokens" not in systems[name]:
        return [], None
    measured = LABELS[name] + ("" if name == best else f", whose notes {LABELS[best]} reuses")
    cfg, chunks = inputs
    from src.data.chunk import hf_token_counter

    count = hf_token_counter(yaml_load(REPO / "configs" / "data.yaml")["tokenizer"])
    eval_dir = next(d for d in run_dirs if d.name == sources[name])
    est = cost.estimate(eval_dir, name, systems[name], chunks, count, cfg, project_gpu_hours())
    lines = ["", "## Cost per article (estimate; no API calls)", "",
             f"Same pipeline, same prompts: {est['prompt_tokens_per_episode']:,} prompt + "
             f"{est['completion_tokens_per_episode']:,} completion tokens per episode (measured for {measured}), "
             f"priced at published per-token rates (OpenRouter, checked {est['prices_checked']}). Self-hosted = "
             f"{est['seconds_per_episode_1xT4']:.0f} T4-seconds per episode (wall time × T4s used) × one T4's "
             "hourly price. "
             "An API model would emit somewhat different token counts, so treat these as estimates.", "",
             "| Option | USD per article | USD per 1,000 articles |", "|---|---|---|"]
    for k, v in est["self_hosted_usd_per_article"].items():
        lines.append(f"| Self-hosted fine-tuned 4B · {k} | ${v:.4f} | ${1000 * v:,.2f} |")
    for k, v in sorted(est["api_usd_per_article"].items(), key=lambda kv: kv[1]):
        lines.append(f"| API · `{k}` | ${v:.4f} | ${1000 * v:,.2f} |")
    if est.get("one_time_gpu_hours") is not None:
        lines += ["", f"One-time cost of building the model: {est['one_time_gpu_hours']} GPU-hours "
                      f"(free on Kaggle; ${est['one_time_usd_at_reference_gpu']:.2f} at the reference GPU price)."]
    return lines, est


def yaml_load(path: Path) -> dict:
    import yaml

    return yaml.safe_load(path.read_text())


def headline(systems: dict, est: dict | None = None) -> str | None:
    best = best_system(systems)
    if not best or "base" not in systems:
        return None
    b, t = systems["base"], systems[best]
    rel = (t["faithfulness"] - b["faithfulness"]) / max(b["faithfulness"], 1e-9)
    halluc = b["hallucinated_entities_per_article"], t["hallucinated_entities_per_article"]
    cut = (halluc[0] - halluc[1]) / halluc[0] if halluc[0] else None
    faith = f"**{abs(100 * rel):.0f}% {'more' if rel >= 0 else 'less'}** faithful"
    entities = ("" if cut is None else f"with **{abs(100 * cut):.0f}% {'fewer' if cut >= 0 else 'more'}** "
                "invented entities per article, ")
    return (f"A QLoRA-fine-tuned 4B model ({LABELS[best]}), trained on Kaggle's free T4 GPUs, writes articles that are "
            f"{faith} to the transcript than the base model "
            f"({pct(b['faithfulness'])} → {pct(t['faithfulness'])} of sentences supported), " + entities
            + f"and an SEO checklist score of **{pct(t['seo_score'])}** (base {pct(b['seo_score'])}), "
            f"at {num(t4_seconds(systems[timed_system(systems, best)]), 0)} T4-seconds per episode"
            + (f", about **{est['times_cheaper_than_comparator']:.0f}× cheaper** per article than "
               f"`{est['comparator']}` via API (estimate)." if est and est.get("times_cheaper_than_comparator") else "."))


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
    split = " (38 TalkRL + 12 Changelog)" if n_eps == 50 else " (smoke subset of the 50)"
    lines = ["# Results", "",
             f"Held-out test set: {n_eps} episodes{split}, frozen before any data generation. "
             "Every system runs the same map-reduce pipeline (chunk notes → merged notes → article + social pack) "
             "with the same prompts and sampling. Numbers come from `results.json` in the runs listed at the bottom.", ""]
    cost_lines, est = cost_section([Path(a) for a in argv], systems, sources)
    if (h := headline(systems, est)):
        lines += ["**Headline:** " + h, ""]
    lines += ["## Main comparison", "",
              "| System | Faithfulness ↑ | Invented entities / article ↓ | Invented numbers / article ↓ | "
              "Articles with any invention ↓ | SEO score ↑ | Flesch (target 50–70) | Valid social JSON ↑ | "
              "X posts ≤ 280 ↑ | Tokens/s | s / episode |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
    for n in names:
        s = systems[n]
        lines.append(f"| {LABELS[n]} | {pct(s['faithfulness'])} | {num(s['hallucinated_entities_per_article'])} | "
                     f"{num(s['hallucinated_numbers_per_article'])} | {pct(s['articles_with_any_hallucination'])} | "
                     f"{pct(s['seo_score'])} | {num(s['flesch'], 1)} ({pct(s['flesch_in_target_rate'])} in range) | "
                     f"{pct(s['social_valid_json_rate'])} | {pct(s['x_posts_ok_rate'])} | {num(s['tokens_per_s'], 0)} | "
                     f"{num(s['seconds_per_episode'], 0)} |")
    gpus = max(systems[n].get("gpus", 1) for n in names)
    lines += ["", f"Generation ran on {gpus} T4{'s (tensor parallel)' if gpus > 1 else ''}; s / episode is wall time."]
    for n in names:
        if (src := systems[n].get("notes_from")) in systems:
            lines.append(f"{LABELS[n]} writes its articles from {LABELS[src]}'s Stage A notes (DPO trained on "
                         "Stage B only), so its tokens/s and s / episode cover Stage B only.")
    checks = list(next(iter(systems.values()))["seo_check_pass_rates"])
    lines += cost_lines
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
