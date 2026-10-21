"""Phase 8 demo: paste a transcript or upload short audio -> SEO article + social pack + scores.

  python app/gradio_app.py --adapter runs/<date>/04_dpo-v<N>/adapter              # NVIDIA GPU (vLLM)
  python app/gradio_app.py --adapter runs/<date>/04_dpo-v<N>/adapter --backend hf  # CPU / Apple GPU (slow)
  python app/gradio_app.py --backend hf                                             # base model, no adapter

On Kaggle: in a GPU notebook with the 04_dpo output attached, run
  !python app/gradio_app.py --adapter /kaggle/input/pod2post-04-dpo/adapter --share
and open the public link it prints. Generation uses exactly the pipeline that was evaluated
(src/infer/pipeline.py), so the demo shows what the reported numbers measure.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import yaml  # noqa: E402

from src.data.chunk import chunk_episode  # noqa: E402
from src.eval.seo import seo_checks, social_checks  # noqa: E402
from src.infer import pipeline  # noqa: E402

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


def make_generator(backend: str, adapter: str | None):
    g = EVAL_CFG["generation"]
    if backend == "vllm":
        from src.infer.generator import VLLMGenerator

        gen = VLLMGenerator(EVAL_CFG["base_model"], max_lora_rank=16 if adapter else 0,
                            max_model_len=g["max_model_len"], gpu_memory_utilization=g["gpu_memory_utilization"],
                            temperature=g["temperature"], top_p=g["top_p"], top_k=g["top_k"])
        gen.use_adapter("demo" if adapter else None, adapter)
        return gen
    return HFGenerator(EVAL_CFG["base_model"], adapter, temperature=g["temperature"])


def transcribe(audio_path: str) -> list[dict]:
    from faster_whisper import WhisperModel

    model = WhisperModel(yaml.safe_load((REPO / "configs" / "asr.yaml").read_text())["model"],
                         device="auto", compute_type="default")
    segments, _ = model.transcribe(audio_path, vad_filter=True)
    return [{"speaker": None, "text": s.text.strip(), "start": round(s.start, 2)} for s in segments]


def text_to_utterances(text: str) -> list[dict]:
    """'Name: words' lines keep their speaker; anything else is one paragraph per line."""
    utts = []
    for line in (x.strip() for x in text.splitlines()):
        if not line:
            continue
        name, sep, rest = line.partition(":")
        if sep and 0 < len(name.split()) <= 4 and rest.strip():
            utts.append({"speaker": name.strip(), "text": rest.strip(), "start": None})
        else:
            utts.append({"speaker": None, "text": line, "start": None})
    return utts


def generate(gen, title: str, transcript: str, audio: str | None, keyword: str) -> tuple[str, str, str, str]:
    utterances = transcribe(audio) if audio else text_to_utterances(transcript or "")
    if not utterances:
        return "Paste a transcript or upload audio first.", "{}", "{}", ""
    episode = {"episode_id": "demo", "split": "demo", "utterances": utterances}
    chunks = [c["text"] for c in chunk_episode(episode, gen.count, DATA_CFG["chunk"]["max_tokens"],
                                               DATA_CFG["chunk"]["overlap_tokens"])]
    g = EVAL_CFG["generation"]
    keywords = {"demo": (keyword.strip().lower(), [])} if keyword.strip() else None
    [r] = pipeline.run([{"episode_id": "demo", "title": title or "Podcast episode", "chunks": chunks}], gen,
                       g["max_tokens"], g["notes_budget_tokens"], keywords=keywords)
    scores = {"primary_keyword": r["primary_keyword"], "chunks": len(chunks),
              "seconds": r["seconds_per_episode"], "social": social_checks(r["social"])}
    if r["article"]:
        seo = seo_checks(r["article"], r["primary_keyword"])
        scores["seo"] = {"score": seo["score"], "failed_checks": [k for k, v in seo["checks"].items() if not v],
                         "words": seo["words"], "keyword_density": seo["density"]}
        try:  # grounding needs spaCy's English model; skip quietly if it is not installed
            from src.data.filter import checkable_text, grounded_entity, ner_entities, norm, ungrounded_numbers

            src = norm("\n".join(chunks))
            text = checkable_text(r["article"], r["social"])
            scores["entities_not_in_transcript"] = [e for e in ner_entities(text) if not grounded_entity(e, src)]
            scores["numbers_not_in_transcript"] = ungrounded_numbers(text, "\n".join(chunks))
        except OSError:
            pass
    article = r["article"] or "*(the model did not produce a parseable article)*\n\n```\n" + r["output"][:3000] + "\n```"
    return article, json.dumps(r["social"] or {}, indent=2, ensure_ascii=False), json.dumps(scores, indent=2), \
        f"**Target keyword:** {r['primary_keyword']}\n\n```\n{r['notes_text']}\n```"


def build_ui(gen):
    import gradio as gr

    with gr.Blocks(title="pod2post") as ui:
        gr.Markdown("# pod2post\nPodcast transcript → SEO article + social pack, with faithfulness and SEO checks.")
        with gr.Row():
            with gr.Column():
                title = gr.Textbox(label="Episode title", placeholder="e.g. Building drone delivery at Zipline")
                keyword = gr.Textbox(label="Target keyword (optional; picked from the notes if empty)")
                with gr.Tab("Transcript"):
                    transcript = gr.Textbox(label="Transcript", lines=16,
                                            placeholder="Speaker: text, one turn per line (or plain paragraphs)")
                with gr.Tab("Audio"):
                    audio = gr.Audio(label="Short audio clip", type="filepath")
                go = gr.Button("Generate", variant="primary")
            with gr.Column():
                with gr.Tab("Article"):
                    article = gr.Markdown()
                with gr.Tab("Social pack"):
                    social = gr.Code(language="json")
                with gr.Tab("Scores"):
                    scores = gr.Code(language="json")
                with gr.Tab("Episode notes"):
                    notes = gr.Markdown()
        go.click(lambda *a: generate(gen, *a), [title, transcript, audio, keyword], [article, social, scores, notes])
    return ui


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--adapter", help="LoRA adapter folder (03_sft or 04_dpo output /adapter)")
    ap.add_argument("--backend", choices=["vllm", "hf"], default="vllm")
    ap.add_argument("--share", action="store_true", help="public Gradio link (needed on Kaggle)")
    args = ap.parse_args()
    build_ui(make_generator(args.backend, args.adapter)).launch(share=args.share)


if __name__ == "__main__":
    main()
