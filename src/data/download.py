"""Phase 1: fetch sources, normalise them to one episode schema, freeze the test set.

  .venv/bin/python -m src.data.download

Writes data/processed/episodes.jsonl (train + test, labelled by "split") and
data/processed/test_ids.sha256. If the hash file already exists the test set is
frozen: a rebuild that would select different test episodes fails instead.

Episode schema: episode_id, source, split, title, url, date, summary, license,
utterances: [{speaker, text, start}]
"""
from __future__ import annotations

import json
import random
import re
import sys
import urllib.request
import zipfile
from pathlib import Path

import yaml

from src.data.chunk import write_jsonl
from src.data.testset import HASH_FILE, id_hash, load_hashes, write_hashes

REPO = Path(__file__).resolve().parents[2]
RAW = REPO / "data" / "raw"
PROCESSED = REPO / "data" / "processed"
CHANGELOG_SPEAKER_RE = re.compile(r"^\*\*(.+?):\*\*\s*(.*)$")


def hf_file(repo: str, filename: str, subdir: str) -> Path:
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(repo, filename, repo_type="dataset", local_dir=RAW / subdir))


def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def mediasum(cfg: dict, seed: int) -> list[dict]:
    strata = [[] for _ in cfg["strata"]]
    with zipfile.ZipFile(hf_file(cfg["repo"], cfg["file"], "mediasum")) as z:
        with z.open(z.namelist()[0]) as f:
            for line in f:
                d = json.loads(line)
                words = sum(len(u.split()) for u in d["utt"])
                for bucket, st in zip(strata, cfg["strata"]):
                    if st["min_words"] <= words < st["max_words"]:
                        bucket.append(d)
    rng = random.Random(seed)
    rows = sorted((d for bucket, st in zip(strata, cfg["strata"])
                   for d in (bucket if st["n"] is None else rng.sample(bucket, st["n"]))), key=lambda d: d["id"])
    return [{
        "episode_id": f"mediasum:{d['id']}", "source": "mediasum", "split": "train",
        "title": d["summary"], "url": d["url"], "date": d["date"], "summary": d["summary"],
        "license": "MediaSum: research use only",
        "utterances": [{"speaker": s, "text": t, "start": None} for s, t in zip(d["speaker"], d["utt"])],
    } for d in rows]


def talkrl(cfg: dict) -> list[dict]:
    import pyarrow.parquet as pq

    rows = [r for r in pq.read_table(hf_file(cfg["repo"], cfg["file"], "talkrl")).to_pylist()
            if r["title"] not in cfg.get("exclude_titles", [])]
    return [{
        "episode_id": f"talkrl:{slugify(r['title'])}", "source": "talkrl", "split": "test",
        "title": r["title"], "url": r["link"], "date": None, "summary": r["summary"],
        "license": "TalkRL: no license stated; evaluation only, not redistributed",
        "utterances": [{"speaker": None, "text": s["text"], "start": s["start"]} for s in r["segments"]],
    } for r in rows]


def parse_changelog(markdown: str) -> list[dict]:
    """'**Name:** text' paragraphs; unlabelled paragraphs continue the previous speaker."""
    utts = []
    for para in (p.strip() for p in markdown.split("\n\n")):
        if not para:
            continue
        m = CHANGELOG_SPEAKER_RE.match(para)
        if m:
            utts.append({"speaker": m[1].strip(), "text": m[2].strip(), "start": None})
        elif utts:
            utts[-1]["text"] += " " + para
    return utts


def changelog(cfg: dict) -> list[dict]:
    episodes = []
    for n in cfg["episodes"]:
        path = RAW / "changelog" / f"the-changelog-{n}.md"
        if not path.is_file():
            path.parent.mkdir(parents=True, exist_ok=True)
            url = f"https://raw.githubusercontent.com/{cfg['repo']}/{cfg['branch']}/podcast/the-changelog-{n}.md"
            with urllib.request.urlopen(url, timeout=30) as resp:
                path.write_bytes(resp.read())
        episodes.append({
            "episode_id": f"changelog:the-changelog-{n}", "source": "changelog", "split": "test",
            "title": f"The Changelog #{n}", "url": f"https://changelog.com/podcast/{n}", "date": None,
            "summary": None, "license": "CC BY-SA 4.0 (thechangelog/transcripts)",
            "utterances": parse_changelog(path.read_text(encoding="utf-8")),
        })
    return episodes


def main() -> int:
    cfg = yaml.safe_load((REPO / "configs" / "data.yaml").read_text())
    PROCESSED.mkdir(parents=True, exist_ok=True)
    train = mediasum(cfg["mediasum"], cfg["seed"])
    test = talkrl(cfg["talkrl"]) + changelog(cfg["changelog"])
    test_ids = [e["episode_id"] for e in test]
    if len(set(test_ids)) != len(test_ids):
        raise SystemExit("duplicate test episode ids")

    hash_path = PROCESSED / HASH_FILE
    if hash_path.is_file():
        if load_hashes(hash_path) != {id_hash(i) for i in test_ids}:
            raise SystemExit(f"{hash_path} is frozen and does not match this test selection; refusing to rebuild")
    else:
        write_hashes(test_ids, hash_path)
    overlap = {e["episode_id"] for e in train} & set(test_ids)
    if overlap:
        raise SystemExit(f"train/test overlap: {sorted(overlap)[:3]}")

    n = write_jsonl(PROCESSED / "episodes.jsonl", train + test)
    for name, eps in (("train", train), ("test", test)):
        words = sorted(sum(len(u["text"].split()) for u in e["utterances"]) for e in eps)
        print(f"{name}: {len(eps)} episodes, words median {words[len(words) // 2]:,} "
              f"min {words[0]:,} max {words[-1]:,}")
    print(f"wrote {n} episodes -> {PROCESSED / 'episodes.jsonl'}; test set frozen in {hash_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
