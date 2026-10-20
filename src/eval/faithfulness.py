"""Faithfulness (spec §6, primary metric): share of article sentences the transcript supports.

Each body sentence is matched to its best transcript windows with BM25, then an NLI
cross-encoder scores window -> sentence. A sentence counts as supported when any of
its top-k windows entails it with p >= threshold.
"""
from __future__ import annotations

import math
import re
from collections import Counter

SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'“])")
TIMESTAMP_RE = re.compile(r"^\[\d{1,2}:\d{2}(?::\d{2})?\]\s*")
MD_RE = re.compile(r"\*\*|__|\*|`|^\s*[-*+]\s+|^\s*\d+\.\s+", re.M)
LINK_RE = re.compile(r"\[([^\]]+)\]\([^)]+\)")
TOKEN_RE = re.compile(r"[a-z0-9]+")
MIN_WORDS = 6


def article_sentences(article: str) -> list[str]:
    """Prose sentences only: headings, the meta line and FAQ questions are skipped."""
    sentences = []
    for line in article.splitlines():
        s = line.strip()
        if not s or s.startswith("#") or s.lower().lstrip("*").startswith("meta description"):
            continue
        s = MD_RE.sub("", LINK_RE.sub(r"\1", s)).strip()
        sentences += [x.strip() for x in SENT_SPLIT_RE.split(s) if len(x.split()) >= MIN_WORDS]
    return sentences


def transcript_windows(chunks: list[str], size: int = 5, stride: int = 3) -> list[str]:
    sents, seen = [], set()
    for chunk in chunks:
        for line in chunk.splitlines():
            for s in SENT_SPLIT_RE.split(TIMESTAMP_RE.sub("", line.strip())):
                if s and s not in seen:  # chunks overlap; keep each sentence once
                    seen.add(s)
                    sents.append(s)
    return [" ".join(sents[i:i + size]) for i in range(0, max(len(sents) - size, 0) + 1, stride)] if sents else []


class BM25:
    def __init__(self, docs: list[str], k1: float = 1.5, b: float = 0.75):
        self.docs = [TOKEN_RE.findall(d.lower()) for d in docs]
        self.tf = [Counter(d) for d in self.docs]
        self.avgdl = sum(map(len, self.docs)) / max(len(self.docs), 1)
        df = Counter(t for d in self.tf for t in d)
        n = len(self.docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}
        self.k1, self.b = k1, b

    def top(self, query: str, k: int) -> list[int]:
        q = set(TOKEN_RE.findall(query.lower()))
        scores = []
        for i, (tf, d) in enumerate(zip(self.tf, self.docs)):
            s = sum(self.idf.get(t, 0) * tf[t] * (self.k1 + 1) /
                    (tf[t] + self.k1 * (1 - self.b + self.b * len(d) / self.avgdl)) for t in q if t in tf)
            scores.append((s, i))
        return [i for _, i in sorted(scores, reverse=True)[:k]]


class NLIScorer:
    def __init__(self, model_name: str, device: str = "cuda", batch_size: int = 32):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, torch_dtype=torch.float16 if device.startswith("cuda") else torch.float32).to(device).eval()
        labels = {v.lower(): int(k) for k, v in self.model.config.id2label.items()}
        self.entail = labels["entailment"]  # label order differs between NLI models
        self.device, self.batch_size = device, batch_size

    def entailment(self, pairs: list[tuple[str, str]]) -> list[float]:
        out = []
        for i in range(0, len(pairs), self.batch_size):
            premises, hyps = zip(*pairs[i:i + self.batch_size])
            enc = self.tok(list(premises), list(hyps), truncation="only_first", max_length=512,
                           padding=True, return_tensors="pt").to(self.device)
            with self.torch.no_grad():
                probs = self.model(**enc).logits.float().softmax(-1)[:, self.entail]
            out += probs.tolist()
        return out


def faithfulness(article: str, chunks: list[str], scorer, top_k: int = 3, threshold: float = 0.5) -> dict:
    sentences = article_sentences(article)
    windows = transcript_windows(chunks)
    if not sentences or not windows:
        return {"score": 0.0, "sentences": len(sentences), "supported": 0, "unsupported": sentences[:3]}
    bm25 = BM25(windows)
    pairs, owner = [], []
    for si, s in enumerate(sentences):
        for wi in bm25.top(s, top_k):
            pairs.append((windows[wi], s))
            owner.append(si)
    best = [0.0] * len(sentences)
    for si, p in zip(owner, scorer.entailment(pairs)):
        best[si] = max(best[si], p)
    supported = [p >= threshold for p in best]
    return {"score": round(sum(supported) / len(sentences), 4), "sentences": len(sentences),
            "supported": sum(supported), "unsupported": [s for s, ok in zip(sentences, supported) if not ok][:3]}
