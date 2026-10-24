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


def build_ui(gen, transcriber=None, models: dict | None = None, daily_limit: int | None = None):
    """daily_limit caps articles per UTC day, so a public site cannot run up the GPU bill."""
    import datetime

    import gradio as gr

    transcriber = transcriber or Transcriber()
    models = models or {BASE_LABEL: "loaded"}
    done_today = {}  # {date: articles started}; one entry per day the process has been up
    with gr.Blocks(title="pod2post") as ui:
        gr.Markdown("# pod2post\nPaste a podcast link or upload an episode, and get an SEO article and a social pack, "
                    "with faithfulness and SEO checks. Only use audio you have the right to republish.")
        with gr.Row():
            with gr.Column():
                with gr.Tab("Podcast link"):
                    url = gr.Textbox(label="Podcast URL", placeholder="https://… (.mp3 link, RSS feed or episode page)")
                    episode = gr.Textbox(label="Episode in feed (optional)",
                                         placeholder="part of the episode title; newest episode if empty")
                with gr.Tab("Audio file"):
                    audio = gr.File(label="Episode audio (mp3, m4a, wav, …)", file_types=["audio", "video"],
                                    type="filepath")
                with gr.Tab("Transcript"):
                    transcript = gr.Textbox(label="Transcript", lines=14,
                                            placeholder="Speaker: text, one turn per line (or plain paragraphs)")
                title = gr.Textbox(label="Episode title (optional; taken from the feed or page if empty)")
                keyword = gr.Textbox(label="Target keyword (optional; picked from the notes if empty)")
                model = gr.Dropdown(list(models), value=next(iter(models)), label="Model",
                                    visible=len(models) > 1)
                go = gr.Button("Create article", variant="primary")
                gr.Markdown("If more than one input is filled in, the link is used first, then the file, "
                            "then the transcript."
                            + (" A *Kaggle GPU* article takes about 12-15 minutes, half of it Kaggle starting up; "
                               "one article is written at a time." if any(str(v).startswith(KAGGLE)
                                                                          for v in models.values()) else ""))
            with gr.Column():
                with gr.Tab("Article"):
                    article = gr.Markdown()
                    download = gr.File(label="Download article (.md)")
                with gr.Tab("Social pack"):
                    social = gr.Code(language="json")
                with gr.Tab("Scores"):
                    scores = gr.Code(language="json")
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
            return out["article"], out["file"], out["social"], out["scores"], out["notes"], out["transcript"], \
                out["title"]

        go.click(run, [title, url, episode, audio, transcript, keyword, model],
                 [article, download, social, scores, notes, used, title])
    return ui


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
    ap.add_argument("--max-file-mb", type=int, default=500, help="largest upload (Cloud Run: 32, its request limit)")
    ap.add_argument("--share", action="store_true", help="public Gradio link (needed on Kaggle)")
    args = ap.parse_args()
    backend = pick_backend(args.backend)
    print(f"[pod2post] backend: {backend}", flush=True)
    gen, models = make_generator(backend, args.adapter, args.gguf, args.lora_gguf)
    if args.kaggle and backend != "kaggle":
        models = KAGGLE_MODELS | models
    ui = build_ui(gen, Transcriber(args.asr_model, args.max_minutes), models, args.daily_limit)
    ui.queue(default_concurrency_limit=1).launch(share=args.share, max_file_size=f"{args.max_file_mb}mb")


if __name__ == "__main__":
    main()
