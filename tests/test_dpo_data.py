import json
import random

from src.data.build_dpo import assemble, inject_hallucination, keyword_stuff
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


def test_assemble_mixes_types_round_robin():
    prompts = [{"episode_id": f"e{i}", "prompt": [], "chosen": "c",
                "rejected": {"hallucination": "h", "keyword_stuffed": "k", "generic": "g"}} for i in range(8)]
    pairs = assemble(prompts, {"e0": "s", "e1": "s", "e2": "s", "e3": "s", "e4": "s", "e5": "s"}, seed=0, n=8)
    kinds = [p["rejected_type"] for p in pairs]
    assert len(pairs) == 8 and set(kinds) == {"sft_failure", "hallucination", "keyword_stuffed", "generic"}
    assert all(p["chosen"][0]["role"] == "assistant" for p in pairs)
