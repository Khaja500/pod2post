"""teacher.run() end to end with a fake model instead of vLLM: stages, retry, resume, test-set guard."""
import json

import pytest

from src.data import teacher
from src.data.chunk import read_jsonl, write_jsonl
from src.data.prompts import STAGE_A_SYSTEM, STAGE_B_SYSTEM
from src.data.testset import write_hashes

NOTES = {"summary": "s", "key_points": ["k1", "k2"], "claims": [], "quotes": [], "entities": [],
         "numbers": [], "keywords": ["drone delivery"]}
ARTICLE = ("<article>\n# Drone delivery\nMeta description: m\n\nbody\n## A\n## B\n## FAQ\n### q\n</article>\n"
           '<social>\n{"linkedin": "l", "x_thread": ["x"], "newsletter": "n", "pull_quotes": ["p"]}\n</social>')


class FakeTeacher:
    calls = []

    def __init__(self, cfg):
        self.generated_tokens = 0
        self.first_a = True

    def count(self, text):
        return len(text.split())

    def chat(self, convs, max_tokens, temperature=None):
        FakeTeacher.calls.append((convs[0][0]["content"][:20], len(convs), temperature))
        out = []
        for c in convs:
            if c[0]["content"] == STAGE_A_SYSTEM:
                out.append("not json" if self.first_a else json.dumps(NOTES))  # first answer fails -> retry
                self.first_a = False
            elif c[0]["content"] == STAGE_B_SYSTEM:
                out.append(ARTICLE)
            else:
                out.append(ARTICLE.replace("body", "generic body"))
        self.generated_tokens += 10 * len(convs)
        return out


@pytest.fixture
def data_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(teacher, "Teacher", FakeTeacher)
    FakeTeacher.calls = []
    d = tmp_path / "data"
    d.mkdir()
    chunks, episodes = [], []
    for e in range(12):
        n = 6 if e < 4 else 2  # 4 long episodes, 8 short
        episodes.append({"episode_id": f"m:{e}", "split": "train", "title": f"Episode {e}"})
        chunks += [{"episode_id": f"m:{e}", "split": "train", "chunk_idx": i, "n_chunks": n, "text": f"chunk {e}.{i}"}
                   for i in range(n)]
    episodes.append({"episode_id": "t:0", "split": "test", "title": "Test"})
    chunks.append({"episode_id": "t:0", "split": "test", "chunk_idx": 0, "n_chunks": 1, "text": "held out"})
    write_jsonl(d / "chunks.jsonl", chunks)
    write_jsonl(d / "episodes.jsonl", episodes)
    write_hashes(["t:0"], d / "test_ids.sha256")
    return d


CFG = {"seed": 1, "open_model": {"model": "fake"}, "targets": {"stage_b": 4}, "overgenerate": 0.5,
       "long_share": 0.25, "long_min_chunks": 5, "generic_rewrites": 2, "notes_budget_tokens": 2500,
       "batch_size": 4, "max_tokens": {"stage_a": 10, "stage_b": 10}, "time_budget_s": 1000,
       "smoke_time_budget_s": 1000}


def test_full_run_produces_all_stages(data_dir, tmp_path):
    out = tmp_path / "out"
    stats = teacher.run(data_dir, out, CFG, smoke=False)
    a = list(read_jsonl(out / "stage_a.jsonl"))
    b = list(read_jsonl(out / "stage_b.jsonl"))
    assert stats["selected_episodes"] == 6 and stats["stage_b"] == 6 and stats["generic"] == 2
    assert all(r["notes"] is not None for r in a) and a[0]["retried"]
    assert any(t == 0.3 for _, _, t in FakeTeacher.calls)                # the retry ran at low temperature
    assert {r["episode_id"] for r in b}.isdisjoint({"t:0"})              # never the test episode
    assert b[0]["primary_keyword"] == "drone delivery" and "Key points:" in b[0]["notes_text"]
    assert sum(r["n_chunks"] >= 5 for r in a) >= 6                       # long episodes are included


def test_resume_skips_finished_records(data_dir, tmp_path):
    first = tmp_path / "first"
    teacher.run(data_dir, first, CFG, smoke=False)
    FakeTeacher.calls = []
    second = tmp_path / "second"
    stats = teacher.run(data_dir, second, CFG, smoke=False, resume_dir=first)
    assert FakeTeacher.calls == [] and stats["stage_b"] == 6


def test_time_budget_stops_cleanly(data_dir, tmp_path):
    stats = teacher.run(data_dir, tmp_path / "out", {**CFG, "time_budget_s": -1}, smoke=False)
    assert stats["stopped_early"] and stats["stage_a"] == 0


def test_smoke_selects_three_episodes(data_dir, tmp_path):
    stats = teacher.run(data_dir, tmp_path / "out", CFG, smoke=True)
    assert stats["selected_episodes"] == 3 and stats["generic"] == 1


def test_format_retry_keeps_the_better_attempt():
    bad = ARTICLE.replace("# Drone delivery", "# " + "Very long title " * 6)

    class Gen:
        def __init__(self):
            self.temps = []

        def chat(self, convs, max_tokens, temperature=None):
            self.temps.append(temperature)
            return [bad, ARTICLE] if temperature is None else [ARTICLE]

    texts, retried = teacher.chat_with_format_retry(Gen(), [[], []], 10)
    assert texts == [ARTICLE, ARTICLE] and retried == [True, False]
