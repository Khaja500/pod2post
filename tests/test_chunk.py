import pytest

from src.data.chunk import (chunk_episode, chunk_units, clean_text, format_utterance, merge_segments,
                            normalize_speaker, split_long)
from src.data.download import parse_changelog


def words(text: str) -> int:
    return len(text.split())


def test_clean_text_removes_fillers_and_stutters():
    assert clean_text("um so uh I I think the the model, um, works") == "so I think the model, works"
    assert clean_text("we had had enough") == "we had had enough"
    assert clean_text("  hmm  ") == ""
    assert clean_text("\\[laughs\\] See [the docs](https://x.io) for \\*that\\*.") == "See the docs for *that*."


def test_merge_segments_joins_mid_sentence_asr_splits():
    segs = [{"speaker": None, "text": " we hooked them up to a browser so they", "start": 4.0},
            {"speaker": None, "text": " could retrieve information.", "start": 9.5},
            {"speaker": None, "text": " Next point.", "start": 12.0}]
    merged = merge_segments(segs, min_words=5)
    assert [m["start"] for m in merged] == [4.0, 12.0]
    assert merged[0]["text"] == "we hooked them up to a browser so they could retrieve information."


def test_speaker_and_timestamp_formatting():
    assert normalize_speaker("ERROL BARNETT, CNN NEWSROOM SHOW HOST") == "Errol Barnett"
    assert normalize_speaker("Jerod Santo") == "Jerod Santo"
    assert format_utterance({"speaker": None, "text": "hello there", "start": 75.4}) == "[01:15] hello there"
    assert format_utterance({"speaker": "BLITZER", "text": "Welcome", "start": 3725}) == "[1:02:05] Blitzer: Welcome"


def test_chunks_respect_budget_and_cover_everything():
    units = [f"Speaker{i % 3}: " + " ".join(f"w{i}_{j}" for j in range(37)) for i in range(120)]
    chunks = chunk_units(units, words, max_tokens=300, overlap_tokens=40)
    assert all(c["n_tokens"] <= 300 for c in chunks)
    covered = set()
    for c in chunks:
        covered.update(range(*c["units"]))
    assert covered == set(range(len(units)))
    starts = [c["units"][0] for c in chunks]
    assert starts == sorted(set(starts))                       # strictly advancing
    for prev, nxt in zip(chunks, chunks[1:]):
        assert nxt["units"][0] < prev["units"][1]              # consecutive chunks overlap
        tail = "\n".join(prev["text"].split("\n")[nxt["units"][0] - prev["units"][0]:])
        assert words(tail) >= 40 and nxt["text"].startswith(tail)


def test_overlong_utterance_is_split():
    long_unit = " ".join(f"Sentence {i} has five words." for i in range(200))   # 1000 words
    pieces = split_long(long_unit, words, 120)
    assert all(words(p) <= 120 for p in pieces) and " ".join(pieces) == long_unit
    no_punct = " ".join(["word"] * 500)
    assert all(words(p) <= 120 for p in split_long(no_punct, words, 120))


def test_short_and_empty_inputs():
    assert chunk_units([], words) == []
    assert len(chunk_units(["one short line"], words)) == 1
    ep = {"episode_id": "x:1", "split": "train", "utterances": [{"speaker": "A", "text": "um", "start": None},
                                                                  {"speaker": "B", "text": "real text", "start": None}]}
    [c] = chunk_episode(ep, words)
    assert c["text"] == "B: real text" and c["n_chunks"] == 1


def test_parse_changelog_markdown():
    md = "**Jerod Santo:** Welcome.\n\n**Guest Person:** Thanks.\n\nSecond paragraph.\n\n**Jerod Santo:** Great."
    utts = parse_changelog(md)
    assert [u["speaker"] for u in utts] == ["Jerod Santo", "Guest Person", "Jerod Santo"]
    assert utts[1]["text"] == "Thanks. Second paragraph."


@pytest.mark.parametrize("max_tokens,overlap", [(50, 0), (50, 25), (1500, 150)])
def test_parameter_grid_terminates(max_tokens, overlap):
    units = [" ".join(["x"] * n) for n in (3, 60, 1, 49, 200, 7) * 5]
    chunks = chunk_units(units, words, max_tokens, overlap)
    assert chunks and all(c["n_tokens"] <= max_tokens for c in chunks)
