import json

import pytest

from src.data.filter import grounded_entity, near_duplicates, norm, stage_a_problems, stage_b_problems, ungrounded_numbers

CHUNK = ("Keenan Wyrobek: We started Zipline almost 12 years ago, delivering blood to hospitals in Rwanda. "
         "Today we serve 4,000 hospitals. As I said, ‘the hard part is the logistics.’")


def notes(**overrides):
    base = {"summary": "Zipline delivers blood in Rwanda.", "key_points": ["Started 12 years ago"],
            "claims": [{"speaker": "Keenan Wyrobek", "claim": "serves 4000 hospitals", "timestamp": None}],
            "quotes": [{"speaker": "Keenan Wyrobek", "quote": "the hard part is the logistics", "timestamp": None}],
            "entities": ["Zipline", "Rwanda", "Keenan Wyrobek"], "numbers": ["4,000 hospitals"], "keywords": ["drone delivery"]}
    return {**base, **overrides}


def test_numbers_and_entities_grounding():
    assert ungrounded_numbers("4000 hospitals in 12 years, 3 lessons", CHUNK) == []
    assert ungrounded_numbers("73% of hospitals", CHUNK) == ["73"]
    src = norm(CHUNK)
    assert grounded_entity("Zipline's", src) and grounded_entity("Mr. Keenan Wyrobek", src)
    assert not grounded_entity("Amazon", src)


def test_stage_a_rules():
    assert stage_a_problems(CHUNK, notes()) == []
    assert stage_a_problems(CHUNK, None) == ["invalid_json"]
    assert "quote_not_verbatim" in stage_a_problems(CHUNK, notes(quotes=[{"speaker": None, "quote": "invented words", "timestamp": None}]))
    assert "ungrounded_entity" in stage_a_problems(CHUNK, notes(entities=["Amazon"]))
    assert "ungrounded_number" in stage_a_problems(CHUNK, notes(numbers=["95% success rate"]))


def test_near_duplicates():
    a = " ".join(f"w{i}" for i in range(200))
    assert near_duplicates([a, a + " extra", "completely different text " * 20]) == {1}


@pytest.fixture(scope="module")
def spacy_available():
    pytest.importorskip("spacy")
    try:
        import spacy
        spacy.load("en_core_web_sm")
    except OSError:
        pytest.skip("en_core_web_sm not installed")


def test_stage_b_flags_invented_entity_and_number(spacy_available):
    social = json.dumps({"linkedin": "Zipline in Rwanda.", "x_thread": ["Zipline delivers blood."],
                         "newsletter": "Zipline.", "pull_quotes": ["the hard part is the logistics"]})
    article = ("# Zipline Drone Delivery\nMeta description: Zipline in Rwanda.\n\nZipline delivers blood.\n\n"
               "## One\ntext\n## Two\ntext\n## FAQ\n### Q?\nA.")
    ok = f"<article>\n{article}\n</article>\n<social>\n{social}\n</social>"
    assert stage_b_problems(ok, "notes", CHUNK, "Title") == []
    bad = ok.replace("Zipline delivers blood.", "Zipline delivers blood, said Jeff Bezos, to 73% of Kenya.")
    problems = stage_b_problems(bad, "notes", CHUNK, "Title")
    assert "ungrounded_entity" in problems and "ungrounded_number" in problems
