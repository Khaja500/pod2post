from src.eval.seo import article_parts, format_problems, keyword_count, parse_output, seo_checks, social_checks

KW = "drone delivery"


def make_article(title="Drone Delivery Lessons From Zipline", meta="How drone delivery reached hospitals.",
                 faq=True, h2=3, extra_mentions=5):
    filler = " ".join(["word"] * 60)
    intro = f"Drone delivery is changing logistics. {filler}"
    sections = [f"## Section {i}{' on drone delivery' if i == 0 else ''}\n\n{filler}" for i in range(h2)]
    extra = [f"More about drone delivery here. {filler}" for _ in range(extra_mentions)]
    faq_lines = ["## FAQ", "### What is drone delivery?", "It is delivery by drone."] if faq else []
    return "\n".join([f"# {title}", f"Meta description: {meta}", "", intro, *sections, *extra, *faq_lines])


def test_parse_output_and_parts():
    text = f"preamble\n<article>\n{make_article()}\n</article>\n<social>\n```json\n" \
           '{"linkedin": "a", "x_thread": ["b"], "newsletter": "c", "pull_quotes": ["d"]}\n```\n</social>'
    article, social = parse_output(text)
    assert social["x_thread"] == ["b"]
    p = article_parts(article)
    assert p["title"].startswith("Drone Delivery") and p["meta"].startswith("How")
    assert p["faq"] and p["faq_questions"] == 1 and len(p["h2"]) == 4


def test_keyword_count_is_phrase_and_case_insensitive():
    assert keyword_count("Drone delivery, drone-delivery and DRONE   DELIVERY; drones delivery", KW) == 2


def test_seo_checks_all_pass_and_each_rule_can_fail():
    good = seo_checks(make_article(), KW)
    assert good["score"] == 1.0, good
    assert not seo_checks(make_article(title="A" * 61 + " drone delivery"), KW)["checks"]["title_len_ok"]
    assert not seo_checks(make_article(meta="m" * 161), KW)["checks"]["meta_len_ok"]
    assert not seo_checks(make_article(faq=False), KW)["checks"]["faq_present"]
    assert not seo_checks(make_article(h2=2, faq=False), KW)["checks"]["min_3_h2"]
    stuffed = make_article() + "\n" + "drone delivery " * 60
    starved = make_article(extra_mentions=0) + "\n" + "word " * 2000
    assert not seo_checks(stuffed, KW)["checks"]["density_ok"]
    assert not seo_checks(starved, KW)["checks"]["density_ok"]
    assert not seo_checks(make_article(title="Lessons From Zipline"), KW)["checks"]["kw_in_title"]


def test_format_problems_and_social():
    social = {"linkedin": "a", "x_thread": ["ok", "x" * 281], "newsletter": "c", "pull_quotes": ["d"]}
    assert "x_post_too_long" in format_problems(make_article(), social)
    assert format_problems(None, None) == ["missing_article"]
    assert "invalid_social_json" in format_problems(make_article(), {"linkedin": "only"})
    assert social_checks({"linkedin": "a", "x_thread": ["ok"], "newsletter": "c", "pull_quotes": []})["x_posts_ok"]
