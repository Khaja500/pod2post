"""Faithfulness, episode scoring and the inference pipeline, with fakes for NLI and the LLM."""
import json

from src.eval.faithfulness import BM25, article_sentences, faithfulness, transcript_windows
from src.eval.run_eval import aggregate, score_episode
from src.infer import pipeline
from tests.test_teacher import ARTICLE, NOTES


class WordOverlapNLI:
    """'Entails' when every content word of the hypothesis is in the premise."""
    def entailment(self, pairs):
        out = []
        for premise, hyp in pairs:
            p = set(premise.lower().replace(".", "").split())
            h = [w for w in hyp.lower().replace(".", "").split() if len(w) > 3]
            out.append(1.0 if all(w in p for w in h) else 0.0)
        return out


CHUNKS = ["[00:01] Ann: Zipline started delivering blood to hospitals in Rwanda twelve years ago. "
          "The drones fly about eighty kilometres. Ann: Logistics was always the hard part."]


def test_article_sentences_skip_headings_and_meta():
    art = "# Title here\nMeta description: m m m m m m\n\n## Head\nZipline started delivering blood to hospitals. Short one.\n- **Logistics** was always the hard part here."
    assert article_sentences(art) == ["Zipline started delivering blood to hospitals.",
                                      "Logistics was always the hard part here."]


def test_bm25_and_windows():
    w = transcript_windows(["One fish. Two fish. Red fish. Blue fish. Old fish. New fish. Fish.", "Two fish. Cats."])
    assert w and "Cats." in w[-1] and w[0].count("Two fish.") == 1
    assert BM25(["apples and pears", "cats and dogs"]).top("dogs", 1) == [1]


def test_faithfulness_counts_supported_sentences():
    art = "Zipline started delivering blood to hospitals in Rwanda.\nZipline also launched rockets to Mars yesterday."
    f = faithfulness(art, CHUNKS, WordOverlapNLI())
    assert f["sentences"] == 2 and f["supported"] == 1 and f["score"] == 0.5


def test_score_and_aggregate_penalise_missing_article():
    article = "# Zipline drone delivery\nMeta description: m\n\nZipline started delivering blood to hospitals in Rwanda.\n## A\n## B\n## FAQ\n### q?\na"
    good = score_episode({"episode_id": "e1", "article": article, "social": json.loads(ARTICLE.split("<social>\n")[1].split("\n</social>")[0]),
                          "primary_keyword": "zipline", "n_chunks": 1}, CHUNKS, WordOverlapNLI())
    missing = score_episode({"episode_id": "e2", "article": None, "social": None, "primary_keyword": "x", "n_chunks": 1},
                            CHUNKS, WordOverlapNLI())
    agg = aggregate([good, missing], {"tokens_per_s": 1.0})
    assert good["faithfulness"] == 1.0 and good["hallucinated_entities"] == 0
    assert agg["article_present_rate"] == 0.5 and agg["faithfulness"] == 0.5 and agg["tokens_per_s"] == 1.0


class FakeGen:
    def __init__(self):
        self.calls = 0

    def count(self, text):
        return len(text.split())

    def chat(self, convs, max_tokens, temperature=None):
        self.calls += 1
        return [json.dumps(NOTES) if "segment" in c[1]["content"].lower() else ARTICLE for c in convs]


def test_pipeline_runs_map_then_reduce():
    eps = [{"episode_id": "e1", "title": "T1", "chunks": ["a", "b"]}, {"episode_id": "e2", "title": "T2", "chunks": ["c"]}]
    gen = FakeGen()
    res = pipeline.run(eps, gen, {"stage_a": 10, "stage_b": 10})
    assert gen.calls == 2 and [r["episode_id"] for r in res] == ["e1", "e2"]
    assert res[0]["primary_keyword"] == "drone delivery" and res[0]["article"].startswith("# Drone")
    assert res[0]["timing"]["episodes"] == 2 and res[1]["stage_a_invalid"] == 0


class Clock:
    """Stands in for the time module in src.eval.kernel; each fake LLM call takes 1,000 s."""
    t = 0.0

    def time(self):
        return self.t


class TimedGen(FakeGen):
    def __init__(self, clock):
        super().__init__()
        self.clock, self.generated_tokens, self.generate_seconds, self.adapters = clock, 0, 0.0, []

    def use_adapter(self, name, path=None):
        self.adapters.append(name)

    def chat(self, convs, max_tokens, temperature=None):
        self.clock.t += 1000
        self.generated_tokens += 10 * len(convs)
        self.generate_seconds += 1.0
        return super().chat(convs, max_tokens, temperature)


def test_eval_stops_at_budget_and_resumes(tmp_path, monkeypatch):
    from src.data.chunk import write_jsonl
    from src.data.testset import HASH_FILE, write_hashes
    from src.eval import kernel

    ids = ["e1", "e2", "e3"]
    data = tmp_path / "data"
    data.mkdir()
    write_jsonl(data / "episodes.jsonl", [{"episode_id": e, "title": f"T {e}", "split": "test"} for e in ids])
    write_jsonl(data / "chunks.jsonl", [{"episode_id": e, "split": "test", "text": CHUNKS[0]} for e in ids])
    write_hashes(ids, data / HASH_FILE)
    cfg = {"time_budget_s": 2500, "smoke_time_budget_s": 2500, "episodes_per_batch": 2,
           "generation": {"max_tokens": {"stage_a": 10, "stage_b": 10}, "notes_budget_tokens": 1800}}
    systems = {"base": None, "sft": "/adapters/sft"}
    clock = Clock()
    monkeypatch.setattr(kernel, "time", clock)

    # Run 1: base takes two batches (2 LLM calls each = 4,000 s), then the budget stops sft before it starts.
    first = kernel.run(data, tmp_path / "run1", cfg, systems, smoke=False, gen=TimedGen(clock), scorer=WordOverlapNLI())
    assert first["stopped_by_time_budget"] and first["pending"] == ["sft"] and list(first["systems"]) == ["base"]
    assert first["systems"]["base"]["episodes"] == 3 and first["systems"]["base"]["generated_tokens"] > 0

    # Run 2 resumes from run 1's outputs: base is carried over untouched, only sft is generated.
    clock.t = 0.0
    gen = TimedGen(clock)
    second = kernel.run(data, tmp_path / "run2", cfg, systems, smoke=False, resume_dir=tmp_path / "run1",
                        gen=gen, scorer=WordOverlapNLI())
    assert not second["stopped_by_time_budget"] and second["pending"] == []
    assert set(gen.adapters) == {"sft"} and second["systems"]["base"] == first["systems"]["base"]
    assert len((tmp_path / "run2" / "outputs" / "sft.jsonl").read_text().splitlines()) == 3
    assert json.loads((tmp_path / "run2" / "results.json").read_text())["systems"].keys() == {"base", "sft"}


def test_pipeline_reuses_notes_and_runs_stage_b_only():
    eps = [{"episode_id": "e1", "title": "T1", "chunks": ["a", "b"]}]
    first = pipeline.run(eps, FakeGen(), {"stage_a": 10, "stage_b": 10})
    gen = FakeGen()
    again = pipeline.run(eps, gen, {"stage_a": 10, "stage_b": 10}, notes_from={"e1": first[0]})
    assert gen.calls == 1 and again[0]["notes_text"] == first[0]["notes_text"]
    assert again[0]["primary_keyword"] == first[0]["primary_keyword"] and again[0]["article"].startswith("# Drone")


def test_eval_system_reuses_notes_of_its_source(tmp_path, monkeypatch):
    from src.data.chunk import write_jsonl
    from src.data.testset import HASH_FILE, write_hashes
    from src.eval import kernel

    ids = ["e1", "e2"]
    data = tmp_path / "data"
    data.mkdir()
    write_jsonl(data / "episodes.jsonl", [{"episode_id": e, "title": f"T {e}", "split": "test"} for e in ids])
    write_jsonl(data / "chunks.jsonl", [{"episode_id": e, "split": "test", "text": CHUNKS[0]} for e in ids])
    write_hashes(ids, data / HASH_FILE)
    cfg = {"time_budget_s": 1e9, "smoke_time_budget_s": 1e9, "episodes_per_batch": 2, "notes_from": {"sft_dpo": "sft"},
           "generation": {"max_tokens": {"stage_a": 10, "stage_b": 10}, "notes_budget_tokens": 1800,
                          "tensor_parallel_size": 2}}
    gen = TimedGen(Clock())
    res = kernel.run(data, tmp_path / "out", cfg, {"sft": "/a/sft", "sft_dpo": "/a/dpo"}, smoke=False,
                     gen=gen, scorer=WordOverlapNLI())
    assert gen.calls == 3  # sft: Stage A + Stage B; sft_dpo: Stage B only
    assert res["systems"]["sft_dpo"]["notes_from"] == "sft" and res["systems"]["sft"]["notes_from"] is None
    assert res["systems"]["sft_dpo"]["gpus"] == 2
    read = lambda n: [json.loads(l) for l in (tmp_path / "out" / "outputs" / f"{n}.jsonl").read_text().splitlines()]
    assert [r["notes_text"] for r in read("sft_dpo")] == [r["notes_text"] for r in read("sft")]
