"""Frozen held-out test set (spec §4.2, rule 5): never train on these episodes.

Test episode IDs are stored only as SHA-256 hashes in data/processed/test_ids.sha256
(git-tracked, also shipped in the Kaggle dataset), and every training entry point
calls assert_no_test_leak() before using data.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable

HASH_FILE = "test_ids.sha256"


def id_hash(episode_id: str) -> str:
    return hashlib.sha256(episode_id.encode("utf-8")).hexdigest()


def write_hashes(episode_ids: Iterable[str], path: Path) -> None:
    path.write_text("".join(f"{h}\n" for h in sorted(id_hash(i) for i in episode_ids)), encoding="utf-8")


def load_hashes(path: Path) -> set[str]:
    hashes = {line.strip() for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()}
    if not hashes:
        raise RuntimeError(f"{path} is empty; the test set must be frozen before any training data is built")
    return hashes


def is_test(episode_id: str, hashes: set[str]) -> bool:
    return id_hash(episode_id) in hashes


def assert_no_test_leak(episode_ids: Iterable[str], path: Path) -> None:
    hashes = load_hashes(path)
    leaked = sorted({i for i in episode_ids if is_test(i, hashes)})
    if leaked:
        raise RuntimeError(f"{len(leaked)} held-out test episode(s) found in training data, e.g. {leaked[:3]}")
