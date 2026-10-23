import json
import random

from src.data.build_dpo import assemble, faithfulness_pair, inject_hallucination, keyword_stuff, sampling_order
from src.data.filter import ungrounded_numbers
from src.eval.seo import seo_checks

KW = "drone delivery"
BODY = " ".join(["Zipline flies blood to remote clinics and the drone delivery model keeps improving."] * 3)
ARTICLE = "\n".join(["# Drone delivery at Zipline", "Meta description: How drone delivery works.", "", BODY,
                     "## How drone delivery works", BODY, "## Lessons", BODY, "## FAQ", "### Is it safe?", "Yes, it is."])
SOCIAL = {"linkedin": "post", "x_thread": ["one", "two"], "newsletter": "n", "pull_quotes": ["q"]}
NOTES = "Speakers: Keenan Wyrobek, Jerod Santo\nKey points:\n- Zipline flies blood"


def test_injected_hallucination_is_ungrounded():
    article, social = inject_hallucination(ARTICLE, SOCIAL, KW, NOTES, random.Random(0))
    assert article != ARTICLE and social["x_thread"][0] != "one"
    assert ungrounded_numbers(article, ARTICLE + NOTES)            # invented figures the source never had
    assert any(name in article for name in ("Keenan Wyrobek", "Jerod Santo"))


def test_keyword_stuffing_fails_density():
    article, _ = keyword_stuff(ARTICLE, SOCIAL, KW, random.Random(0))
    assert not seo_checks(article, KW)["checks"]["density_ok"]


def cand(text, faith, seo=1.0, words=800, problems=()):
    return {"text": text, "faithfulness": faith, "seo_score": seo, "words": words, "problems": list(problems)}


def test_faithfulness_pair_prefers_the_most_faithful_clean_answer():
    scored = {"teacher": cand("t", 0.50), "sft": [cand("s1", 0.30), cand("s2", 0.60)]}
    chosen, rejected = faithfulness_pair(scored, margin=0.08)
    assert (chosen["text"], rejected["text"]) == ("s2", "s1")
    scored["sft"][1]["problems"] = ["ungrounded_entity"]                     # a failing sample is never chosen
    assert faithfulness_pair(scored, 0.08)[0]["text"] == "t"
    assert faithfulness_pair({"teacher": cand("t", 0.35), "sft": [cand("s1", 0.30)]}, 0.08) is None   # margin
    assert faithfulness_pair({"teacher": cand("t", 0.9, seo=0.75), "sft": [cand("s1", 0.3)]}, 0.08) is None  # SEO
    assert faithfulness_pair({"teacher": cand("t", 0.9, words=500), "sft": [cand("s1", 0.3)]}, 0.08) is None  # length


def test_assemble_adds_synthetic_pairs_as_a_share():
    prompts = [{"episode_id": f"e{i}", "prompt": [], "chosen": "c",
                "rejected": {"hallucination": "h", "keyword_stuffed": "k", "generic": "g"}} for i in range(12)]
    order = [p["episode_id"] for p in sampling_order(prompts, seed=0, n=12)]
    scored = {e: {"teacher": cand("t", 0.6), "sft": [cand("s1", 0.3), cand("s2", 0.4)]} for e in order[:4]}
    pairs = assemble(prompts, scored, seed=0, n=12, margin=0.08, synthetic_share=0.33)
    kinds = [p["rejected_type"] for p in pairs]
    assert kinds[:4] == ["sft_less_faithful"] * 4 and len(pairs) == 6     # 4 on-policy + 2 synthetic (a third)
    assert set(kinds[4:]) == {"hallucination", "keyword_stuffed"} and pairs[0]["faithfulness"] == [0.6, 0.3]
    assert [p["episode_id"] for p in pairs[:4]] == order[:4] and all(p["chosen"][0]["role"] == "assistant" for p in pairs)
    assert {p["rejected_type"] for p in assemble(prompts, {}, seed=0, n=12)} == {"hallucination", "keyword_stuffed", "generic"}
    one = {order[0]: scored[order[0]]}                                  # smoke: 1 scored prompt, still >= min_pairs
    assert len(assemble(prompts, one, seed=0, n=12)) == 1 and len(assemble(prompts, one, seed=0, n=12, min_pairs=8)) == 8


def test_score_candidate_uses_the_eval_scorer():
    from src.train.score_samples import score_candidate
    from tests.test_eval import CHUNKS, WordOverlapNLI
    from tests.test_teacher import ARTICLE

    prompt = {"episode_id": "e1", "primary_keyword": "drone delivery", "notes_text": NOTES}
    good = score_candidate(ARTICLE, prompt, CHUNKS, WordOverlapNLI())
    assert set(good) == {"text", "problems", "faithfulness", "seo_score", "words", "flesch",
                         "hallucinated_entities", "hallucinated_numbers"} and good["text"] == ARTICLE
    repaired = score_candidate(ARTICLE.replace('["p"]}', '["p"],}'), prompt, CHUNKS, WordOverlapNLI())
    assert "social_not_strict_json" in repaired["problems"]  # the training filter repairs it; the eval would not
