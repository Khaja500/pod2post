import json

from src.data.chunk import write_jsonl
from src.eval import report


def fake_run(path, systems):
    (path / "outputs").mkdir(parents=True)
    (path / "scores").mkdir()
    res = {"episodes": 2, "systems": {}}
    for name, faith in systems.items():
        res["systems"][name] = {
            "episodes": 2, "article_present_rate": 1.0, "faithfulness": faith,
            "hallucinated_entities_per_article": 2.0 - faith, "hallucinated_numbers_per_article": 0.5,
            "articles_with_any_hallucination": 0.5, "seo_score": 0.75, "seo_check_pass_rates": {"kw_in_title": 1.0},
            "flesch": 58.0, "flesch_in_target_rate": 1.0, "social_valid_json_rate": 1.0, "x_posts_ok_rate": 1.0,
            "stage_a_invalid_rate": 0.0, "words": 800, "tokens_per_s": 500.0, "seconds_per_episode": 30.0}
        eps = ["talkrl:a", "changelog:b"]
        write_jsonl(path / "outputs" / f"{name}.jsonl", [{"episode_id": e, "title": e, "primary_keyword": "kw",
                                                          "article": f"# {name} article\nbody words"} for e in eps])
        write_jsonl(path / "scores" / f"{name}.jsonl", [{"episode_id": e, "faithfulness": faith, "seo_score": 0.75,
                                                         "hallucinated_entities": 1} for e in eps])
    (path / "results.json").write_text(json.dumps(res))
    return path


def test_report_combines_runs_and_writes_headline(tmp_path, monkeypatch):
    monkeypatch.setattr(report, "REPO", tmp_path)
    base = fake_run(tmp_path / "r1", {"base": 0.5})
    final = fake_run(tmp_path / "r2", {"sft": 0.7, "sft_dpo": 0.8})
    assert report.main([str(base), str(final)]) == 0
    md = (tmp_path / "reports" / "results.md").read_text()
    assert "| Base (zero-shot) | 50.0%" in md and "| SFT + DPO | 80.0%" in md
    assert "**+60%** more faithful" in md and "### talkrl:a" in md
