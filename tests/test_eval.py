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
