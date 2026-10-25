"""Phase 8 website: podcast URL, audio file or transcript -> SEO article + social pack + scores.

  python app/gradio_app.py                                                          # Mac: llama.cpp + models/*.gguf
  python app/gradio_app.py --adapter runs/<date>/04_dpo-v<N>/adapter              # NVIDIA GPU (vLLM)
  python app/gradio_app.py --adapter <sft+dpo>/adapter --adapter <sft>/adapter    # vLLM, with a model picker
  python app/gradio_app.py --adapter runs/<date>/04_dpo-v<N>/adapter --backend hf  # transformers (slow)

--backend auto (the default) picks vLLM on an NVIDIA GPU, else llama.cpp when llama-server and
the 4-bit GGUF in models/ exist (README: "Run it on a Mac"), else transformers. With llama.cpp
every models/*-lora-*.gguf adapter is loaded, and the page has a model picker.

On Kaggle: in a GPU notebook with the 04_dpo output attached, run
  !python app/gradio_app.py --adapter /kaggle/input/pod2post-04-dpo/adapter --share
and open the public link it prints. Generation uses exactly the pipeline that was evaluated
(src/infer/pipeline.py), so the site shows what the reported numbers measure.

A podcast URL can be a direct audio link, an RSS feed (newest episode, or the one named in
"Episode in feed") or an episode page yt-dlp can read (src/asr/url_audio.py). Audio is
transcribed with faster-whisper: configs/asr.yaml's model on a GPU, --asr-model on CPU.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import yaml  # noqa: E402

from src.infer.article import (  # noqa: E402,F401  (re-exported: the tests and the Kaggle backend use them)
    Transcriber, generate, text_to_utterances, utterances_text)

EVAL_CFG = yaml.safe_load((REPO / "configs" / "eval.yaml").read_text())
DATA_CFG = yaml.safe_load((REPO / "configs" / "data.yaml").read_text())


class HFGenerator:
    """transformers + PEFT fallback for machines without vLLM (CPU or Apple GPU); one prompt at a time."""

    def __init__(self, model: str, adapter: str | None, temperature: float = 0.3):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        self.tok = AutoTokenizer.from_pretrained(model)
        self.model = AutoModelForCausalLM.from_pretrained(
            model, torch_dtype=torch.float16 if device != "cpu" else torch.float32).to(device)
        if adapter:
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(self.model, adapter)
        self.model.eval()
        self.device, self.temperature, self.torch = device, temperature, torch

    def count(self, text: str) -> int:
        return len(self.tok.encode(text, add_special_tokens=False))

    def chat(self, conversations, max_tokens, temperature=None):
        outs = []
        for conv in conversations:
            ids = self.tok.apply_chat_template(conv, add_generation_prompt=True, return_tensors="pt").to(self.device)
            temp = self.temperature if temperature is None else temperature
            with self.torch.no_grad():
                gen = self.model.generate(ids, max_new_tokens=max_tokens, do_sample=temp > 0,
                                          temperature=max(temp, 1e-5), top_p=0.8, top_k=20)
            outs.append(self.tok.decode(gen[0, ids.shape[1]:], skip_special_tokens=True))
        return outs


MODELS = REPO / "models"
BASE_GGUF = MODELS / "Qwen3-4B-Instruct-2507-Q4_K_M.gguf"
BASE_LABEL = "Base Qwen3-4B (more faithful in our eval)"
KAGGLE = "kaggle:"  # models value prefix: the job runs on Kaggle's GPUs (scripts/kaggle_infer.py)
# YouTube asks data-centre machines (Kaggle, Cloud Run) to "sign in to confirm you're not a bot", so
# yt-dlp cannot download there (07_infer-v5); a Kaggle job with such a link would fail after its start-up.
BLOCKED_ON_KAGGLE = re.compile(r"^https?://([a-z0-9-]+\.)*(youtube\.com|youtu\.be)(/|$)", re.I)
KAGGLE_MODELS = {"Kaggle GPU: SFT + DPO (fastest for long episodes)": KAGGLE + "sft_dpo",
                 "Kaggle GPU: SFT": KAGGLE + "sft", "Kaggle GPU: base Qwen3-4B": KAGGLE + "base"}


def adapter_label(path: str) -> str:
    name = "/".join(Path(path).parts[-2:]).lower()  # trained adapters sit in .../04_dpo-v4/adapter
    return "Fine-tuned: SFT + DPO (best SEO)" if "dpo" in name else "Fine-tuned: SFT" if "sft" in name \
        else f"Adapter: {Path(path).stem}"


def pick_backend(backend: str) -> str:
    if backend != "auto":
        return backend
    if shutil.which("nvidia-smi"):
        return "vllm"
    return "llamacpp" if shutil.which("llama-server") and BASE_GGUF.is_file() else "hf"


def make_generator(backend: str, adapters: list[str] | None, gguf: str | None = None,
                   lora_ggufs: list[str] | None = None):
    """Returns the generator and the models the page can switch between: {label: adapter path or None}."""
    adapters = adapters or []
    if backend == "kaggle":  # no model here: every job runs on Kaggle (a small server for the public site)
        return None, dict(KAGGLE_MODELS)
    g = EVAL_CFG["generation"]
    sampling = dict(temperature=g["temperature"], top_p=g["top_p"], top_k=g["top_k"])
    if backend == "llamacpp":
        from src.data.chunk import hf_token_counter
        from src.infer.generator import LlamaCppGenerator

        loras = lora_ggufs if lora_ggufs is not None else [str(p) for p in sorted(MODELS.glob("*-lora-*.gguf"))]
        gen = LlamaCppGenerator(gguf or str(BASE_GGUF), loras, count=hf_token_counter(DATA_CFG["tokenizer"]),
                                ctx=g["max_model_len"], **sampling)
        models = {adapter_label(p): p for p in sorted(loras, key=lambda p: "dpo" not in p)} | {BASE_LABEL: None}
        return gen, models
    if backend == "vllm":
        from src.infer.generator import VLLMGenerator

        gen = VLLMGenerator(EVAL_CFG["base_model"], max_lora_rank=16 if adapters else 0,
                            max_model_len=g["max_model_len"], gpu_memory_utilization=g["gpu_memory_utilization"],
                            **sampling)
        return gen, {adapter_label(p): p for p in sorted(adapters, key=lambda p: "dpo" not in p)} | {BASE_LABEL: None}
    adapter = adapters[0] if adapters else None
    gen = HFGenerator(EVAL_CFG["base_model"], adapter, temperature=g["temperature"])
    return gen, {adapter_label(adapter) if adapter else BASE_LABEL: "loaded"}  # one model, no switching


def kaggle_generate(title: str, transcript: str, audio: str | None, keyword: str, url: str, episode: str,
                    model: str, progress=None, runner=None) -> dict:
    """One episode on Kaggle (07_infer): the same generate() on 2 T4s, ~10 min of start-up included.
    An uploaded file goes to Kaggle as a private dataset first (scripts/kaggle_infer.py upload_audio)."""
    url = (url or "").strip()
    if not url and not audio and not (transcript or "").strip():
        raise ValueError("Paste a podcast link, upload an audio file or paste a transcript first.")
    if BLOCKED_ON_KAGGLE.match(url):
        raise ValueError("YouTube blocks downloads from cloud servers, so this link cannot be fetched. Download the "
                         "episode's audio and upload it in the Audio file tab, or paste the podcast's RSS feed or "
                         "mp3 link instead.")
    if runner is None:
        sys.path.insert(0, str(REPO / "scripts"))
        from kaggle_infer import run_jobs as runner
    job = {"id": "j1", "url": url, "episode": episode or "", "transcript": "" if url or audio else transcript,
           "title": title or "", "keyword": keyword or "", "model": model}
    if audio and not url:  # same priority as generate(): link, then file, then transcript
        job["audio_path"] = audio

    def on_status(status, elapsed_s):
        if progress:
            progress(min(0.9, 0.05 + elapsed_s / 1800), desc=f"Kaggle GPU: {status}, {elapsed_s / 60:.0f} min "
                                                              "(start-up alone takes ~10 min)")
    try:
        res = runner([job], on_status)["j1"]
    except SystemExit as e:  # kaggle_run.py's die() is meant for a terminal; here it must not stop the site
        raise RuntimeError(f"Kaggle: {e}") from None
    if "error" in res:
        raise RuntimeError(f"Kaggle: {res['error']}")
    return res


SAMPLE_TITLE = "Why cities are planting tiny forests"
SAMPLE_TRANSCRIPT = """Host: Welcome back to Green Streets. Today I'm talking with Maya, an urban ecologist who plants tiny forests in cities.
Maya: Thanks for having me. A tiny forest is a dense patch of native trees, about the size of a tennis court.
Host: Why so dense?
Maya: We plant three trees per square metre, following the Miyawaki method from Japan. The trees compete for light, so they grow fast.
Host: How fast are we talking?
Maya: In our sites in Leeds, the trees grew about one metre a year for the first three years. After three years we stop watering and weeding; the forest looks after itself.
Host: And what do neighbourhoods get out of it?
Maya: Shade, cooler streets in summer, and a lot more insects and birds. Our volunteers counted over 500 species of animals and plants at one school site.
Host: Who does the planting?
Maya: Mostly local people. A planting day takes about a hundred volunteers and one weekend. Kids love it, and they come back to check on their trees.
Host: What's the biggest mistake people make?
Maya: Picking the wrong species. You need trees that are native to your area, and you need to prepare the soil properly before planting.
Host: If a listener wants to start one, what should they do first?
Maya: Find a piece of land, talk to the council, and then talk to your neighbours. A tiny forest only works if the community wants it.
Host: Maya, thank you.
Maya: Thank you."""

THEME_FONT = "Inter"
CSS = """
.gradio-container { max-width: 1240px !important; margin: 0 auto !important; }
#hero { padding: 28px 30px 22px; border-radius: 18px; color: #fff;
        background: linear-gradient(120deg, #4338ca 0%, #7c3aed 55%, #db2777 100%); margin-bottom: 6px; }
#hero h1 { margin: 0 0 6px; font-size: 2.1rem; font-weight: 800; letter-spacing: -0.02em; color: #fff; }
#hero p { margin: 0; font-size: 1.02rem; opacity: 0.92; color: #fff; max-width: 760px; }
#hero .steps { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 16px; }
#hero .steps span { background: rgba(255,255,255,0.16); border: 1px solid rgba(255,255,255,0.28);
                    padding: 5px 12px; border-radius: 999px; font-size: 0.86rem; color: #fff; }
.section-label { font-weight: 700; font-size: 0.95rem; margin: 2px 0 -4px; opacity: 0.85; }
.note { font-size: 0.86rem; opacity: 0.78; }
#go { font-size: 1.05rem; font-weight: 700; min-height: 48px; }
.stats { display: grid; grid-template-columns: repeat(auto-fit, minmax(120px, 1fr)); gap: 10px; }
.stat { border: 1px solid var(--border-color-primary); background: var(--block-background-fill);
        border-radius: 12px; padding: 10px 12px; }
.stat .v { font-size: 1.35rem; font-weight: 800; line-height: 1.2; }
.stat .k { font-size: 0.78rem; opacity: 0.7; text-transform: uppercase; letter-spacing: 0.04em; }
.stat.wide { grid-column: span 2; } .stat.wide .v { font-size: 1.05rem; padding-top: 4px; }
.stat.good .v { color: #16a34a; } .stat.warn .v { color: #d97706; } .stat.bad .v { color: #dc2626; }
.empty { border: 2px dashed var(--border-color-primary); border-radius: 14px; padding: 36px 24px;
         text-align: center; opacity: 0.75; }
#article { min-height: 320px; }
footer { display: none !important; }
"""

EMPTY_ARTICLE = ('<div class="empty"><b>Your article will appear here.</b><br>Add an episode on the left and press '
                 '<i>Create article</i>. No episode handy? Press <i>Try a sample transcript</i>.</div>')


def stat_cards(scores_json: str) -> str:
    """The Scores JSON as a row of cards: SEO, length, keyword, social, unsupported facts, time."""
    import html

    try:
        s = json.loads(scores_json or "{}")
    except ValueError:
        return ""
    seo, social, src = s.get("seo") or {}, s.get("social") or {}, s.get("source") or {}
    unsupported = len(s.get("entities_not_in_transcript") or []) + len(s.get("numbers_not_in_transcript") or [])
    cards = []
    if "score" in seo:
        pct = round(100 * seo["score"])
        cards.append((f"{pct}%", "SEO checks", "good" if pct >= 80 else "warn" if pct >= 60 else "bad"))
        cards.append((f"{seo.get('words', 0):,}", "words", ""))
    if s.get("primary_keyword"):
        cards.append((html.escape(str(s["primary_keyword"])), "keyword", "wide"))
    if social:
        cards.append((str(social.get("x_posts", 0)), "X posts", "good" if social.get("x_posts_ok") else "warn"))
    if "entities_not_in_transcript" in s:
        cards.append((str(unsupported), "unsupported facts", "good" if unsupported == 0 else "warn"))
    if src.get("audio_minutes"):
        cards.append((f"{src['audio_minutes']:.0f} min", "audio", ""))
    if s.get("generation_seconds") is not None:
        cards.append((f"{s['generation_seconds'] / 60:.1f} min", "writing time", ""))
    return '<div class="stats">' + "".join(f'<div class="stat {c}"><div class="v">{v}</div><div class="k">{k}</div>'
                                            "</div>" for v, k, c in cards) + "</div>"


def social_markdown(social_json: str) -> str:
    """The social pack as text people can copy: LinkedIn, the X thread, the newsletter blurb, pull quotes."""
    try:
        s = json.loads(social_json or "{}")
    except ValueError:
        return f"```\n{social_json}\n```"
    if not isinstance(s, dict) or not s:
        return "*The model did not produce a social pack.*"
    parts = []
    if s.get("linkedin"):
        parts.append(f"### LinkedIn\n{s['linkedin']}")
    if isinstance(s.get("x_thread"), list) and s["x_thread"]:
        n = len(s["x_thread"])
        parts.append("### X thread\n" + "\n".join(f"{i}. {t}  \n   <sub>{i}/{n} · {len(t)} characters</sub>"
                                                  for i, t in enumerate(s["x_thread"], 1)))
    if s.get("newsletter"):
        parts.append(f"### Newsletter blurb\n{s['newsletter']}")
    if isinstance(s.get("pull_quotes"), list) and s["pull_quotes"]:
        parts.append("### Pull quotes\n" + "\n\n".join(f"> {q}" for q in s["pull_quotes"]))
    return "\n\n".join(parts) or f"```json\n{social_json}\n```"


def build_ui(gen, transcriber=None, models: dict | None = None, daily_limit: int | None = None):
    """daily_limit caps articles per UTC day, so a public site cannot run up the GPU bill."""
    import datetime

    import gradio as gr

    transcriber = transcriber or Transcriber()
    models = models or {BASE_LABEL: "loaded"}
    on_kaggle = any(str(v).startswith(KAGGLE) for v in models.values())
    done_today = {}  # {date: articles started}; one entry per day the process has been up
    with gr.Blocks(title="pod2post: podcast to article") as ui:
        gr.HTML('<div id="hero"><h1>pod2post</h1><p>Turn a podcast episode into an SEO article and a ready-to-post '
                "social pack, written by a fine-tuned Qwen3-4B and checked against the transcript.</p>"
                '<div class="steps"><span>1 · Add an episode</span><span>2 · Pick a model</span>'
                "<span>3 · Create the article</span></div></div>")
        with gr.Row(equal_height=False):
            with gr.Column(scale=5):
                gr.Markdown("1 · Your episode", elem_classes="section-label")
                with gr.Tabs() as sources:
                    with gr.Tab("Podcast link", id="link"):
                        url = gr.Textbox(label="Podcast URL", placeholder="https://… (.mp3 link, RSS feed or "
                                                                          "episode page)")
                        episode = gr.Textbox(label="Episode in feed (optional)",
                                             placeholder="part of the episode title; newest episode if empty")
                    with gr.Tab("Audio file", id="file"):
                        audio = gr.File(label="Episode audio (mp3, m4a, wav, …)", file_types=["audio", "video"],
                                        type="filepath")
                    with gr.Tab("Transcript", id="text"):
                        transcript = gr.Textbox(label="Transcript", lines=12, max_lines=24,
                                                placeholder="Speaker: text, one turn per line (or plain paragraphs)")
                        sample = gr.Button("Try a sample transcript", size="sm", variant="secondary")
                gr.Markdown("2 · Model and options", elem_classes="section-label")
                model = gr.Dropdown(list(models), value=next(iter(models)), label="Model",
                                    visible=len(models) > 1)
                with gr.Accordion("Title and keyword (optional)", open=False):
                    title = gr.Textbox(label="Episode title", placeholder="taken from the feed or page if empty")
                    keyword = gr.Textbox(label="Target keyword", placeholder="picked from the notes if empty")
                with gr.Row():
                    go = gr.Button("Create article", variant="primary", elem_id="go", scale=3)
                    clear = gr.Button("Clear", variant="secondary", scale=1)
                gr.Markdown("If more than one input is filled in, the link is used first, then the file, then the "
                            "transcript. Only use audio you have the right to republish."
                            + (" A *Kaggle GPU* article takes about 12–15 minutes, half of it Kaggle starting up; "
                               "one article is written at a time." if on_kaggle else ""), elem_classes="note")
            with gr.Column(scale=7):
                stats = gr.HTML()
                with gr.Tabs():
                    with gr.Tab("Article"):
                        article = gr.Markdown(EMPTY_ARTICLE, elem_id="article")
                        download = gr.File(label="Download article (.md)", visible=False)
                    with gr.Tab("Social pack"):
                        social_view = gr.Markdown("*Appears with the article.*")
                        with gr.Accordion("JSON", open=False):
                            social = gr.Code(language="json", show_label=False)
                    with gr.Tab("Scores"):
                        scores = gr.Code(language="json", show_label=False)
                    with gr.Tab("Episode notes"):
                        notes = gr.Markdown()
                    with gr.Tab("Transcript used"):
                        used = gr.Textbox(lines=16, show_label=False)

        def run(title_, url_, episode_, audio_, transcript_, keyword_, model_, progress=gr.Progress()):
            today = datetime.datetime.now(datetime.timezone.utc).date()
            if daily_limit and done_today.get(today, 0) >= daily_limit:
                raise gr.Error(f"This site's limit of {daily_limit} articles a day has been reached. "
                               "Please try again tomorrow (UTC).")
            done_today.update({today: done_today.get(today, 0) + 1})  # failed runs count too: they used the GPU
            # fields in a closed accordion are never rendered, so Gradio sends None for them, not ""
            title_, url_, episode_, transcript_, keyword_ = (x or "" for x in (title_, url_, episode_, transcript_,
                                                                                keyword_))
            choice = models.get(model_)
            try:
                if isinstance(choice, str) and choice.startswith(KAGGLE):
                    out = kaggle_generate(title_, transcript_, audio_, keyword_, url_, episode_,
                                          choice[len(KAGGLE):], progress)
                else:
                    if choice != "loaded":  # the queue runs one request at a time, so switching is safe
                        gen.use_adapter(model_ if choice else None, choice)
                    out = generate(gen, title_, transcript_, audio_, keyword_, url=url_, episode=episode_,
                                   transcriber=transcriber, progress=progress)
            except (ValueError, RuntimeError) as e:
                raise gr.Error(str(e)) from None
            except Exception as e:  # network errors, undecodable audio, yt-dlp failures
                raise gr.Error(f"{type(e).__name__}: {e}") from None
            return out["article"], gr.File(out["file"], visible=bool(out["file"])), out["social"], out["scores"], \
                out["notes"], out["transcript"], out["title"], stat_cards(out["scores"]), \
                social_markdown(out["social"])

        go.click(run, [title, url, episode, audio, transcript, keyword, model],
                 [article, download, social, scores, notes, used, title, stats, social_view])
        # filled in the browser, so a quick "Create article" click right after already sends the sample
        sample.click(None, None, [transcript, title, url],
                     js=f"() => [{json.dumps(SAMPLE_TRANSCRIPT)}, {json.dumps(SAMPLE_TITLE)}, '']")
        clear.click(lambda: ("", "", None, "", "", "", EMPTY_ARTICLE, gr.File(None, visible=False), "", "",
                             "*Appears with the article.*", "", "", gr.Tabs(selected="link")), None,
                    [url, episode, audio, transcript, title, keyword, article, download, stats, scores, social_view,
                     social, used, sources])
    return ui


def theme():
    import gradio as gr

    return gr.themes.Soft(primary_hue="indigo", secondary_hue="violet", neutral_hue="slate",
                          font=[gr.themes.GoogleFont(THEME_FONT), "ui-sans-serif", "system-ui", "sans-serif"])


def serve_http2(ui, port: int, max_file_mb: int) -> None:
    """Serve over cleartext HTTP/2 with Hypercorn. Cloud Run caps HTTP/1 requests at 32 MB (one upload is
    one request) but not HTTP/2 ones (`--use-http2` in deploy.sh), and uvicorn, Gradio's own server, speaks
    HTTP/1 only."""
    import asyncio

    import gradio as gr
    from fastapi import FastAPI
    from hypercorn.asyncio import serve
    from hypercorn.config import Config

    app = gr.mount_gradio_app(FastAPI(), ui, path="/", max_file_size=f"{max_file_mb}mb", theme=theme(),
                              css=CSS)
    config = Config()
    config.bind = [f"0.0.0.0:{port}"]
    print(f"[pod2post] serving HTTP/2 (h2c) on port {port}", flush=True)
    asyncio.run(serve(app, config))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--adapter", action="append", help="vLLM / transformers: LoRA adapter folder (03_sft or 04_dpo "
                                                        "output /adapter); repeat for a model picker (vLLM)")
    ap.add_argument("--backend", choices=["auto", "vllm", "llamacpp", "hf", "kaggle"], default="auto",
                    help="kaggle: no local model, every article is written on Kaggle's GPUs")
    ap.add_argument("--gguf", help=f"llama.cpp: base model GGUF (default {BASE_GGUF.relative_to(REPO)})")
    ap.add_argument("--lora-gguf", action="append", help="llama.cpp: adapter GGUF, repeatable "
                                                          "(default: every models/*-lora-*.gguf)")
    ap.add_argument("--asr-model", default="small", help="faster-whisper model when there is no GPU (default: small)")
    ap.add_argument("--max-minutes", type=float, default=180, help="longest audio accepted")
    ap.add_argument("--daily-limit", type=int, help="most articles per UTC day (a public site's cost cap)")
    ap.add_argument("--kaggle", action="store_true", help="also offer the models on Kaggle's GPUs (needs the kaggle "
                                                           "CLI signed in and `scripts/kaggle_infer.py adapters`)")
    ap.add_argument("--max-file-mb", type=int, default=500, help="largest upload (Cloud Run over HTTP/1: 32)")
    ap.add_argument("--http2", action="store_true", help="serve HTTP/2 with Hypercorn on $PORT (Cloud Run "
                                                          "--use-http2: no 32 MB upload limit)")
    ap.add_argument("--share", action="store_true", help="public Gradio link (needed on Kaggle)")
    args = ap.parse_args()
    backend = pick_backend(args.backend)
    print(f"[pod2post] backend: {backend}", flush=True)
    gen, models = make_generator(backend, args.adapter, args.gguf, args.lora_gguf)
    if args.kaggle and backend != "kaggle":
        models = KAGGLE_MODELS | models
    ui = build_ui(gen, Transcriber(args.asr_model, args.max_minutes), models, args.daily_limit)
    ui.queue(default_concurrency_limit=1)
    if args.http2:
        serve_http2(ui, int(os.environ.get("PORT", 8080)), args.max_file_mb)
    else:
        ui.launch(share=args.share, max_file_size=f"{args.max_file_mb}mb", theme=theme(), css=CSS)


if __name__ == "__main__":
    main()
