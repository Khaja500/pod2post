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
    out = gradio_app.generate(FakeGen(), "Zipline", TRANSCRIPT, None, "")
    assert out["article"].startswith("# Drone") and json.loads(out["social"])["x_thread"] == ["x"]
    s = json.loads(out["scores"])
    assert s["primary_keyword"] == "drone delivery" and "seo" in s and s["chunks"] == 1
    assert out["file"].endswith("zipline.md") and Path(out["file"]).read_text().startswith("# Drone")
    assert out["transcript"].splitlines()[0] == "Ann: Zipline delivers blood to hospitals in Rwanda."


def test_generate_from_a_url_downloads_transcribes_and_takes_the_feed_title(tmp_path):
    calls = []

    def fetch(url, dest, episode=""):
        calls.append((url, episode))
        return {"path": tmp_path / "audio.mp3", "title": "Drones in Rwanda", "kind": "feed", "audio_url": "https://a/x.mp3"}

    def transcriber(path):
        assert path == str(tmp_path / "audio.mp3")
        return [{"speaker": None, "text": u["text"], "start": 0.0} for u in gradio_app.text_to_utterances(TRANSCRIPT)], 12.5

    out = gradio_app.generate(FakeGen(), "", "ignored transcript", None, "", url=" https://feed ", episode="rwanda",
                              transcriber=transcriber, fetch=fetch)
    assert calls == [(" https://feed ", "rwanda")] and out["title"] == "Drones in Rwanda"
    s = json.loads(out["scores"])
    assert s["source"]["kind"] == "feed" and s["source"]["audio_minutes"] == 12.5
    assert "ignored" not in out["transcript"]


def test_generate_without_input_raises():
    with pytest.raises(ValueError):
        gradio_app.generate(FakeGen(), "", "  ", None, "")


def test_build_ui():
    pytest.importorskip("gradio")
    assert gradio_app.build_ui(FakeGen()) is not None


def test_adapter_labels_read_the_run_folder():
    assert gradio_app.adapter_label("runs/2026-10-23/04_dpo-v4/adapter") == "Fine-tuned: SFT + DPO (best SEO)"
    assert gradio_app.adapter_label("/app/adapters/sft") == "Fine-tuned: SFT"
    assert gradio_app.adapter_label("models/pod2post-sft-dpo-lora-f16.gguf").startswith("Fine-tuned: SFT + DPO")


def test_site_switches_adapters_by_name_and_stops_at_the_daily_limit():
    gr = pytest.importorskip("gradio")
    class SwitchingGen(FakeGen):
        adapters = []

        def use_adapter(self, name, path=None):
            self.adapters.append(name)

    gen = SwitchingGen()
    models = {"Fine-tuned: SFT + DPO (best SEO)": "/a/dpo", "Fine-tuned: SFT": "/a/sft", gradio_app.BASE_LABEL: None}
    ui = gradio_app.build_ui(gen, models=models, daily_limit=3)
    run = next(f.fn for f in ui.fns.values() if f.name == "run")
    for label in ("Fine-tuned: SFT", "Fine-tuned: SFT + DPO (best SEO)"):
        assert run("Zipline", "", "", None, TRANSCRIPT, "", label)[0].startswith("# Drone")
    assert gen.adapters == ["Fine-tuned: SFT", "Fine-tuned: SFT + DPO (best SEO)"]  # distinct names: vLLM caches by name
    assert run(None, None, None, None, TRANSCRIPT, None, gradio_app.BASE_LABEL)[0].startswith("# Drone")  # unrendered fields
    with pytest.raises(gr.Error, match="limit of 3"):
        run("Zipline", "", "", None, TRANSCRIPT, "", gradio_app.BASE_LABEL)


def test_kaggle_models_send_one_job_and_show_its_result():
    sent = []

    def runner(jobs, on_status):
        sent.extend(jobs)
        on_status("running", 120)
        return {"j1": gradio_app.generate(FakeGen(), "Zipline", TRANSCRIPT, None, "")}

    out = gradio_app.kaggle_generate("Zipline", "", None, "", " https://feed ", "rwanda", "sft", runner=runner)
    assert sent == [{"id": "j1", "url": "https://feed", "episode": "rwanda", "transcript": "", "title": "Zipline",
                     "keyword": "", "model": "sft"}] and out["article"].startswith("# Drone")
    sent.clear()
    gradio_app.kaggle_generate("", "ignored", "/tmp/a.mp3", "", "", "", "sft", runner=runner)
    assert sent[0]["audio_path"] == "/tmp/a.mp3" and sent[0]["transcript"] == ""  # the upload goes to Kaggle
    with pytest.raises(ValueError, match="Paste a podcast link"):
        gradio_app.kaggle_generate("", " ", None, "", "", "", "sft", runner=runner)
    with pytest.raises(RuntimeError, match="Kaggle: boom"):
        gradio_app.kaggle_generate("", TRANSCRIPT, None, "", "", "", "base", runner=lambda jobs, cb: {"j1": {"error": "boom"}})


def test_a_kaggle_driver_exit_becomes_an_error_not_a_crash():
    def runner(jobs, on_status):
        raise SystemExit("unknown kernel")

    with pytest.raises(RuntimeError, match="unknown kernel"):
        gradio_app.kaggle_generate("", TRANSCRIPT, None, "", "", "", "base", runner=runner)


def test_youtube_links_are_refused_before_a_kaggle_run_starts():
    def runner(jobs, on_status):
        raise AssertionError("no Kaggle run for a link that cannot be downloaded there")

    for url in ("https://www.youtube.com/watch?v=QigAo0NKUIM", "https://youtu.be/QigAo0NKUIM",
                "https://m.youtube.com/watch?v=x"):
        with pytest.raises(ValueError, match="YouTube blocks downloads"):
            gradio_app.kaggle_generate("", "", None, "", url, "", "sft_dpo", runner=runner)
    assert not gradio_app.BLOCKED_ON_KAGGLE.match("https://notyoutube.com/feed")


def test_result_cards_and_social_pack_render_from_the_json():
    out = gradio_app.generate(FakeGen(), "Zipline", TRANSCRIPT, None, "")
    cards = gradio_app.stat_cards(out["scores"])
    assert "SEO checks" in cards and "drone delivery" in cards and "X posts" in cards
    assert gradio_app.stat_cards("not json") == ""
    md = gradio_app.social_markdown(json.dumps({"linkedin": "Post", "x_thread": ["one", "two"], "newsletter": "N",
                                                "pull_quotes": ["Q"]}))
    assert "### LinkedIn\nPost" in md and "1. one" in md and "2/2" in md and "> Q" in md
    assert "did not produce" in gradio_app.social_markdown("{}")


def test_sample_transcript_parses_into_speaker_turns():
    utts = gradio_app.text_to_utterances(gradio_app.SAMPLE_TRANSCRIPT)
    assert len(utts) > 10 and {u["speaker"] for u in utts} == {"Host", "Maya"}
