import json

from src.data.notes import merge_notes, parse_notes, pick_keywords


def words(text):
    return len(text.split())


def chunk_notes(i, n_points=6):
    return {"summary": f"Segment {i} discusses topic {i} in depth with examples.",
            "key_points": [f"point {i}.{k} about subject{i} detail{k} " + "x " * 8 for k in range(n_points)],
            "claims": [{"speaker": "Ann", "claim": f"claim {i} " + "y " * 10, "timestamp": f"0{i}:00"}],
            "quotes": [{"speaker": "Ann", "quote": f"quote {i} " + "z " * 10, "timestamp": None}],
            "entities": [f"Entity{i}", "Shared Org"], "numbers": [f"{i}0 percent of users"],
            "keywords": ["drone delivery", f"topic {i}"]}


def test_parse_notes_lenient_and_strict():
    good = json.dumps(chunk_notes(1))
    assert parse_notes(f"Here you go:\n```json\n{good}\n```")["entities"] == ["Entity1", "Shared Org"]
    assert parse_notes('{"summary": "x"}') is None
    assert parse_notes("not json") is None


def test_pick_keywords_by_frequency():
    primary, secondary = pick_keywords([chunk_notes(i) for i in range(4)])
    assert primary == "drone delivery" and secondary == ["topic 0", "topic 1", "topic 2"]


def test_merge_respects_budget_and_keeps_every_chunk():
    notes = [chunk_notes(i) for i in range(20)]
    merged = merge_notes(notes, words, budget=900)
    assert words(merged) <= 900
    for i in range(20):
        assert f"point {i}.0" in merged                     # top key point of every chunk survives
    assert merged.count("Shared Org") == 1                  # entities deduplicated
    full = merge_notes(notes[:2], words, budget=10_000)
    assert "point 1.5" in full and "Claims:" in full and "Quotes:" in full


def test_sanitize_drops_unverifiable_quotes_entities_and_figures():
    from src.data.notes import sanitize_notes
    chunk = "Ann: We serve 4,000 hospitals in Rwanda. The hard part is logistics."
    n = {"summary": "s", "key_points": ["serves 4000 hospitals", "grew 73% last year"],
         "claims": [{"speaker": "Ann", "claim": "costs fell 40%", "timestamp": None}],
         "quotes": [{"speaker": "Ann", "quote": "The hard part is logistics", "timestamp": None},
                    {"speaker": "Ann", "quote": "We love Kenya", "timestamp": None}],
         "entities": ["Rwanda", "Kenya"], "numbers": ["4,000 hospitals", "73% growth"], "keywords": ["k"]}
    s = sanitize_notes(n, chunk)
    assert [q["quote"] for q in s["quotes"]] == ["The hard part is logistics"]
    assert s["entities"] == ["Rwanda"] and s["numbers"] == ["4,000 hospitals"]
    assert s["key_points"] == ["serves 4000 hospitals"] and s["claims"] == []
