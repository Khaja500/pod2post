"""The demo's wiring with a fake model: transcript in, article + social + scores out."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
import gradio_app  # noqa: E402

from tests.test_eval import FakeGen  # noqa: E402

TRANSCRIPT = "Ann: Zipline delivers blood to hospitals in Rwanda.\nBob: How far do the drones fly?\nAnn: About eighty kilometres."


def test_text_to_utterances_keeps_speakers():
    utts = gradio_app.text_to_utterances(TRANSCRIPT + "\nA plain paragraph without a speaker label, longer than four words: yes.")
    assert [u["speaker"] for u in utts] == ["Ann", "Bob", "Ann", None]


def test_generate_returns_article_social_and_scores():
    article, social, scores, notes = gradio_app.generate(FakeGen(), "Zipline", TRANSCRIPT, None, "")
    assert article.startswith("# Drone") and json.loads(social)["x_thread"] == ["x"]
    s = json.loads(scores)
    assert s["primary_keyword"] == "drone delivery" and "seo" in s and s["chunks"] == 1


def test_build_ui():
    pytest.importorskip("gradio")
    assert gradio_app.build_ui(FakeGen()) is not None
