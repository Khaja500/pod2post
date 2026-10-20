"""WER spot-check (Phase 2): a Whisper transcript vs a human reference transcript.

  .venv/bin/python -m src.asr.wer runs/<date>/01_transcribe-v<N>/transcripts/changelog-661.json

The Changelog reference is lightly edited (fillers and false starts removed), so
the number is an upper bound on the true WER.
"""
from __future__ import annotations

import json
import re
import sys
import urllib.request
from pathlib import Path

from src.data.chunk import FILLER_RE, MD_ESCAPE_RE, MD_LINK_RE, STAGE_RE

SPEAKER_MD_RE = re.compile(r"\*\*[^*]+:\*\*")
NON_WORD_RE = re.compile(r"[^a-z0-9' ]+")


def normalize(text: str) -> list[str]:
    text = STAGE_RE.sub(" ", MD_LINK_RE.sub(r"\1", MD_ESCAPE_RE.sub(r"\1", SPEAKER_MD_RE.sub(" ", text))))
    text = re.sub(r"\[[^\]]*\]", " ", text)                 # remaining bracketed notes
    text = FILLER_RE.sub(" ", text.lower().replace("’", "'"))
    text = NON_WORD_RE.sub(" ", text.replace("-", " "))
    return [w.strip("'") for w in text.split() if w.strip("'")]


def wer(reference: list[str], hypothesis: list[str]) -> dict:
    import jiwer

    out = jiwer.process_words(" ".join(reference), " ".join(hypothesis))
    return {"wer": round(out.wer, 4), "substitutions": out.substitutions, "deletions": out.deletions,
            "insertions": out.insertions, "ref_words": len(reference), "hyp_words": len(hypothesis)}


def main(argv: list[str]) -> int:
    import yaml

    if len(argv) != 1:
        print(__doc__)
        return 1
    repo = Path(__file__).resolve().parents[2]
    cfg = yaml.safe_load((repo / "configs" / "asr.yaml").read_text())["wer_reference"]
    hyp_ep = json.loads(Path(argv[0]).read_text(encoding="utf-8"))
    with urllib.request.urlopen(cfg["transcript"], timeout=60) as resp:
        reference = resp.read().decode("utf-8")
    result = wer(normalize(reference), normalize(" ".join(u["text"] for u in hyp_ep["utterances"])))
    print(json.dumps({"episode": cfg["id"], "model": hyp_ep.get("asr", {}).get("model"), **result}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
