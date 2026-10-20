"""Transcript cleaning and ~1.5k-token chunking with overlap (spec §3, Stage A input).

  .venv/bin/python -m src.data.chunk    # data/processed/episodes.jsonl -> chunks.jsonl

Chunks break only between utterances where possible; an utterance longer than a
chunk is split at sentence boundaries, then at word boundaries as a last resort.
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Callable, Iterable

TokenCounter = Callable[[str], int]

FILLER_RE = re.compile(r"\b(?:u+m+|u+h+|e+rm|e+r|a+h+|h+m+|m+h*m+)\b[,.]?\s*", re.I)
REPEAT_RE = re.compile(r"\b(\w+)(?:\s+\1\b)+", re.I)
KEEP_REPEATS = {"had", "that"}  # "had had" / "that that" are often grammatical
SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
MD_ESCAPE_RE = re.compile(r"\\([\[\]()*_#`\\-])")
MD_LINK_RE = re.compile(r"\[([^\]]+)\]\([^)]+\)")
STAGE_RE = re.compile(r"\[(?:laughs?|laughter|chuckles?|inaudible|crosstalk|music|applause|sighs?)\]\s*", re.I)


def clean_text(text: str) -> str:
    """Drop fillers, stutters, markdown escapes and stage directions; normalise whitespace.

    Casing is left alone: Whisper and the transcript sources are already cased, and
    ASR segments start mid-sentence, so re-capitalising them makes things worse.
    """
    text = STAGE_RE.sub("", MD_LINK_RE.sub(r"\1", MD_ESCAPE_RE.sub(r"\1", text)))
    text = FILLER_RE.sub("", text)
    text = REPEAT_RE.sub(lambda m: m[0] if m[1].lower() in KEEP_REPEATS else m[1], text)
    return re.sub(r"\s+([,.!?;:])", r"\1", re.sub(r"\s+", " ", text)).strip()


def merge_segments(utterances: list[dict], min_words: int = 40, max_words: int = 200) -> list[dict]:
    """Join consecutive same-speaker segments (e.g. Whisper's mid-sentence splits) into
    paragraphs that end a sentence, keeping the first segment's timestamp."""
    out: list[dict] = []
    for u in utterances:
        text = u["text"].strip()
        if not text:
            continue
        if out and out[-1]["speaker"] == u.get("speaker"):
            prev = out[-1]
            n = len(prev["text"].split())
            if not ((prev["text"].endswith((".", "!", "?")) and n >= min_words) or n >= max_words):
                prev["text"] += " " + text
                continue
        out.append({"speaker": u.get("speaker"), "text": text, "start": u.get("start")})
    return out


def normalize_speaker(name: str | None) -> str | None:
    """'ERROL BARNETT, CNN NEWSROOM SHOW HOST' -> 'Errol Barnett'."""
    if not name or not name.strip():
        return None
    name = name.split(",", 1)[0].strip()
    return name.title() if name.isupper() else name


def format_timestamp(seconds: float) -> str:
    s = int(seconds)
    return f"{s // 3600}:{s % 3600 // 60:02d}:{s % 60:02d}" if s >= 3600 else f"{s // 60:02d}:{s % 60:02d}"


def format_utterance(utt: dict) -> str:
    text = clean_text(utt["text"])
    if not text:
        return ""
    prefix = f"[{format_timestamp(utt['start'])}] " if utt.get("start") is not None else ""
    speaker = normalize_speaker(utt.get("speaker"))
    return f"{prefix}{speaker}: {text}" if speaker else f"{prefix}{text}"


def _pack(pieces: list[str], count: TokenCounter, max_tokens: int, sep: str) -> list[str]:
    out, cur = [], ""
    for piece in pieces:
        cand = f"{cur}{sep}{piece}" if cur else piece
        if cur and count(cand) > max_tokens:
            out.append(cur)
            cur = piece
        else:
            cur = cand
    if cur:
        out.append(cur)
    return out


def split_long(unit: str, count: TokenCounter, max_tokens: int) -> list[str]:
    """Split one over-long utterance into pieces of at most max_tokens."""
    if count(unit) <= max_tokens:
        return [unit]
    pieces = []
    for sent in _pack(SENTENCE_RE.split(unit), count, max_tokens, " "):
        pieces += _pack(sent.split(" "), count, max_tokens, " ") if count(sent) > max_tokens else [sent]
    return pieces


def chunk_units(units: list[str], count: TokenCounter, max_tokens: int = 1500,
                overlap_tokens: int = 150) -> list[dict]:
    """Greedily pack utterance lines into chunks; each chunk repeats the tail of the previous one."""
    units = [p for u in units if u for p in split_long(u, count, max_tokens - overlap_tokens)]
    sizes = [count(u) + 1 for u in units]  # +1 for the joining newline
    chunks, start = [], 0
    while start < len(units):
        end, total = start, 0
        while end < len(units) and total + sizes[end] <= max_tokens:
            total += sizes[end]
            end += 1
        end = max(end, start + 1)
        text = "\n".join(units[start:end])
        chunks.append({"text": text, "n_tokens": count(text), "units": [start, end]})
        if end >= len(units):
            break
        back, overlap = end, 0
        while back - 1 > start and overlap < overlap_tokens:
            back -= 1
            overlap += sizes[back]
        start = back
    return chunks


def chunk_episode(episode: dict, count: TokenCounter, max_tokens: int = 1500, overlap_tokens: int = 150) -> list[dict]:
    units = [format_utterance(u) for u in merge_segments(episode["utterances"])]
    chunks = chunk_units(units, count, max_tokens, overlap_tokens)
    return [{"episode_id": episode["episode_id"], "split": episode["split"], "chunk_idx": i,
             "n_chunks": len(chunks), "n_tokens": c["n_tokens"], "text": c["text"]}
            for i, c in enumerate(chunks)]


def hf_token_counter(repo: str) -> TokenCounter:
    """Exact token count with the base model's tokenizer (downloaded once, ungated)."""
    from huggingface_hub import hf_hub_download
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(hf_hub_download(repo, "tokenizer.json"))
    return lambda text: len(tok.encode(text, add_special_tokens=False).ids)


def read_jsonl(path: Path) -> Iterable[dict]:
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path: Path, rows: Iterable[dict]) -> int:
    n = 0
    with Path(path).open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


def main() -> None:
    import yaml

    repo = Path(__file__).resolve().parents[2]
    cfg = yaml.safe_load((repo / "configs" / "data.yaml").read_text())
    processed = repo / "data" / "processed"
    count = hf_token_counter(cfg["tokenizer"])
    rows = [c for ep in read_jsonl(processed / "episodes.jsonl")
            for c in chunk_episode(ep, count, cfg["chunk"]["max_tokens"], cfg["chunk"]["overlap_tokens"])]
    n = write_jsonl(processed / "chunks.jsonl", rows)
    per_ep = Counter((r["split"], r["episode_id"]) for r in rows)
    for split in ("train", "test"):
        counts = sorted(v for (s, _), v in per_ep.items() if s == split)
        if counts:
            print(f"{split}: {len(counts)} episodes, {sum(counts)} chunks, "
                  f"chunks/episode median {counts[len(counts) // 2]} max {counts[-1]}")
    print(f"wrote {n} chunks -> {processed / 'chunks.jsonl'}; max chunk {max(r['n_tokens'] for r in rows)} tokens")


if __name__ == "__main__":
    sys.exit(main())
