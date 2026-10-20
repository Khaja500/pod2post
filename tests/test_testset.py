from pathlib import Path

import pytest

from src.data.testset import HASH_FILE, assert_no_test_leak, load_hashes, write_hashes

REPO = Path(__file__).resolve().parents[1]


def test_leak_is_detected(tmp_path):
    path = tmp_path / HASH_FILE
    write_hashes(["talkrl:a", "changelog:b"], path)
    assert_no_test_leak(["mediasum:CNN-1", "mediasum:NPR-2"], path)
    with pytest.raises(RuntimeError, match="held-out"):
        assert_no_test_leak(["mediasum:CNN-1", "changelog:b"], path)


def test_empty_hash_file_refuses(tmp_path):
    (tmp_path / HASH_FILE).write_text("")
    with pytest.raises(RuntimeError, match="frozen"):
        load_hashes(tmp_path / HASH_FILE)


@pytest.mark.skipif(not (REPO / "data" / "processed" / HASH_FILE).is_file(), reason="test set not built yet")
def test_frozen_test_set_has_50_episodes():
    assert len(load_hashes(REPO / "data" / "processed" / HASH_FILE)) == 50
