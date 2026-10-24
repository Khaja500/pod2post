"""Podcast URL, audio file or transcript -> article + social pack + scores: what the website
(app/gradio_app.py) and the Kaggle inference kernel (kaggle/07_infer) both run, so a job gives the same
result wherever it runs. Generation is the evaluated pipeline (src/infer/pipeline.py)."""
from __future__ import annotations

import json
import re
import shutil
import tempfile
import time
from pathlib import Path

import yaml

from src.data.chunk import chunk_episode
from src.eval.seo import seo_checks, social_checks
from src.infer import pipeline

REPO = Path(__file__).resolve().parents[2]
EVAL_CFG = yaml.safe_load((REPO / "configs" / "eval.yaml").read_text())
DATA_CFG = yaml.safe_load((REPO / "configs" / "data.yaml").read_text())
ASR_CFG = yaml.safe_load((REPO / "configs" / "asr.yaml").read_text())


class Transcriber:
    """faster-whisper, loaded on first use. A GPU runs the batched pipeline with configs/asr.yaml
    (on the last GPU, so vLLM keeps GPU 0); a CPU runs `cpu_model` in int8."""

    def __init__(self, cpu_model: str = "small", max_minutes: float = 180):
        self.cpu_model, self.max_minutes, self.pipe = cpu_model, max_minutes, None

    def _load(self):
        from faster_whisper import BatchedInferencePipeline, WhisperModel

        try:
            import torch  # loads cuDNN/cuBLAS for ctranslate2 when present
            gpus = torch.cuda.device_count()
        except ImportError:
            gpus = 0
        if gpus:
            model = WhisperModel(ASR_CFG["model"], device="cuda", device_index=gpus - 1,
                                 compute_type=ASR_CFG["compute_type"])
            self.pipe, self.batch = BatchedInferencePipeline(model), ASR_CFG["batch_size"]
        else:
            self.pipe, self.batch = WhisperModel(self.cpu_model, device="cpu", compute_type="int8"), None

    def __call__(self, audio_path: str) -> tuple[list[dict], float]:
        from faster_whisper import decode_audio

        if self.pipe is None:
            self._load()
        audio = decode_audio(str(audio_path), sampling_rate=16000)
        minutes = len(audio) / 16000 / 60
        if minutes > self.max_minutes:
            raise ValueError(f"The audio is {minutes:.0f} minutes long; the limit is {self.max_minutes:.0f}.")
        kwargs = dict(beam_size=ASR_CFG["beam_size"], language=ASR_CFG["language"], vad_filter=ASR_CFG["vad_filter"])
        if self.batch:
            kwargs["batch_size"] = self.batch
        segments, _ = self.pipe.transcribe(audio, **kwargs)
        return [{"speaker": None, "text": s.text.strip(), "start": round(s.start, 2)} for s in segments], minutes


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


def utterances_text(utterances: list[dict]) -> str:
    return "\n".join(f"{u['speaker']}: {u['text']}" if u["speaker"] else u["text"] for u in utterances)


def _no_progress(*_, **__):
    return None


def generate(gen, title: str, transcript: str, audio: str | None, keyword: str, url: str = "", episode: str = "",
             transcriber=None, fetch=None, progress=_no_progress) -> dict:
    """The first input given wins: podcast URL, then audio file, then pasted transcript.
    Returns the article, social pack, scores, notes, transcript, title and a Markdown file."""
    t0, source = time.time(), {}
    work = Path(tempfile.mkdtemp(prefix="pod2post-"))
    if url and url.strip():
        if fetch is None:
            from src.asr.url_audio import resolve as fetch
        progress(0.05, desc="Downloading the episode")
        got = fetch(url, work / "download", episode=episode or "")
        audio, title = str(got["path"]), title or got["title"] or ""
        source = {"url": url.strip(), "kind": got["kind"], "audio_url": got["audio_url"]}
    if audio:
        progress(0.2, desc="Transcribing (long episodes take a while)")
        t1 = time.time()
        try:
            utterances, minutes = (transcriber or Transcriber())(audio)
        finally:
            shutil.rmtree(work / "download", ignore_errors=True)  # a downloaded episode is ~50-100 MB
        source.update(audio_minutes=round(minutes, 1), transcribe_seconds=round(time.time() - t1))
    else:
        utterances = text_to_utterances(transcript or "")
    if not utterances:
        raise ValueError("Paste a podcast link, upload an audio file or paste a transcript first.")
    episode_row = {"episode_id": "demo", "split": "demo", "utterances": utterances}
    chunks = [c["text"] for c in chunk_episode(episode_row, gen.count, DATA_CFG["chunk"]["max_tokens"],
                                               DATA_CFG["chunk"]["overlap_tokens"])]
    progress(0.5, desc=f"Writing notes on {len(chunks)} part(s), then the article")
    g = EVAL_CFG["generation"]
    keywords = {"demo": (keyword.strip().lower(), [])} if keyword.strip() else None
    [r] = pipeline.run([{"episode_id": "demo", "title": title or "Podcast episode", "chunks": chunks}], gen,
                       g["max_tokens"], g["notes_budget_tokens"], keywords=keywords)
    progress(0.95, desc="Checking the article")
    scores = {"primary_keyword": r["primary_keyword"], "chunks": len(chunks), "source": source or None,
              "generation_seconds": r["seconds_per_episode"], "total_seconds": round(time.time() - t0),
              "social": social_checks(r["social"])}
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
        except (ImportError, OSError):
            pass
    article = r["article"] or "*(the model did not produce a parseable article)*\n\n```\n" + r["output"][:3000] + "\n```"
    md = None
    if r["article"]:
        slug = re.sub(r"[^a-z0-9]+", "-", (title or r["primary_keyword"] or "article").lower()).strip("-")[:60]
        md = work / f"{slug or 'article'}.md"
        md.write_text(r["article"] + "\n", encoding="utf-8")
    return {"article": article, "social": json.dumps(r["social"] or {}, indent=2, ensure_ascii=False),
            "scores": json.dumps(scores, indent=2), "transcript": utterances_text(utterances), "title": title,
            "notes": f"**Target keyword:** {r['primary_keyword']}\n\n```\n{r['notes_text']}\n```",
            "file": str(md) if md else None}
