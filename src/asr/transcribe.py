"""Phase 2: podcast audio -> timestamped transcript with faster-whisper.

  .venv/bin/python -m src.asr.transcribe fetch   # local: download the audio in configs/asr.yaml
                                                 #        to data/raw/audio/ (+ audio_manifest.json)
  transcribe_all(...)                            # on Kaggle, called by kaggle/01_transcribe

Each transcript is written in the episode schema used by src/data (utterances with
start times, no speakers), so it can go straight into chunking and the pipeline.
"""
from __future__ import annotations

import json
import shutil
import sys
import time
import urllib.request
from pathlib import Path

SAMPLE_RATE = 16000


def fetch(cfg: dict, dest: Path) -> list[dict]:
    dest.mkdir(parents=True, exist_ok=True)
    manifest = []
    for item in [*cfg["episodes"], cfg["wer_reference"]]:
        path = dest / f"{item['id']}.mp3"
        if not path.is_file():
            print(f"downloading {item['id']} ...", flush=True)
            req = urllib.request.Request(item["url"], headers={"User-Agent": "pod2post/0.1"})
            with urllib.request.urlopen(req, timeout=120) as resp, open(path, "wb") as f:
                shutil.copyfileobj(resp, f)
        manifest.append({"id": item["id"], "file": path.name, "title": item["title"], "url": item["url"],
                         "license": item["license"], "bytes": path.stat().st_size})
    (dest / "audio_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def transcribe_all(audio_dir: Path, out_dir: Path, cfg: dict, smoke: bool) -> list[dict]:
    """Transcribe every manifest entry (SMOKE: first smoke_seconds of the first one)."""
    import torch  # loads cuDNN/cuBLAS into the process so ctranslate2 finds them

    from faster_whisper import BatchedInferencePipeline, WhisperModel, decode_audio

    assert torch.cuda.is_available(), "01_transcribe needs a GPU"
    model = WhisperModel(cfg["model"], device="cuda", compute_type=cfg["compute_type"])
    pipe = BatchedInferencePipeline(model)
    manifest = json.loads((audio_dir / "audio_manifest.json").read_text(encoding="utf-8"))
    out_dir.mkdir(parents=True, exist_ok=True)
    report = []
    for item in manifest[:1] if smoke else manifest:
        audio = decode_audio(str(audio_dir / item["file"]), sampling_rate=SAMPLE_RATE)
        if smoke:
            audio = audio[: cfg["smoke_seconds"] * SAMPLE_RATE]
        duration = len(audio) / SAMPLE_RATE
        t0 = time.time()
        segments, info = pipe.transcribe(audio, batch_size=cfg["batch_size"], beam_size=cfg["beam_size"],
                                         language=cfg["language"], vad_filter=cfg["vad_filter"])
        segments = [{"start": round(s.start, 2), "end": round(s.end, 2), "text": s.text.strip()} for s in segments]
        elapsed = time.time() - t0
        asr = {"model": cfg["model"], "duration_s": round(duration, 1), "elapsed_s": round(elapsed, 1),
               "rtf": round(elapsed / duration, 4), "language_probability": round(info.language_probability, 3),
               "segments": len(segments), "words": sum(len(s["text"].split()) for s in segments)}
        episode = {
            "episode_id": f"cc:{item['id']}", "source": "cc_audio", "split": "realworld",
            "title": item["title"], "url": item["url"], "date": None, "summary": None, "license": item["license"],
            "utterances": [{"speaker": None, "text": s["text"], "start": s["start"]} for s in segments],
            "asr": asr,
        }
        (out_dir / f"{item['id']}.json").write_text(json.dumps(episode, indent=1, ensure_ascii=False), encoding="utf-8")
        print(f"{item['id']}: {duration / 60:.1f} min audio in {elapsed:.0f}s (RTF {asr['rtf']}), "
              f"{asr['words']} words", flush=True)
        report.append({"id": item["id"], **asr})
    return report


def main(argv: list[str]) -> int:
    import yaml

    repo = Path(__file__).resolve().parents[2]
    cfg = yaml.safe_load((repo / "configs" / "asr.yaml").read_text())
    if argv[:1] == ["fetch"]:
        manifest = fetch(cfg, repo / "data" / "raw" / "audio")
        print(f"{len(manifest)} files, {sum(m['bytes'] for m in manifest) / 1e6:.0f} MB in data/raw/audio/")
        return 0
    print(__doc__)
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
