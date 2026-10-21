from src.data.chunk import write_jsonl
from src.eval import judge

CHUNKS = ["Ann: Zipline delivers blood to hospitals in Rwanda. The drones fly eighty kilometres. "
          "Logistics was always the hard part. We started twelve years ago."]


def test_verdict_parsing_and_order_combination():
    assert judge.verdict("Winner: **B** because...") == "b" and judge.verdict("no idea") is None
    assert judge.combine("a", "b") == "win"      # system preferred in both orders
    assert judge.combine("a", "tie") == "win"
    assert judge.combine("b", "a") == "loss"
    assert judge.combine("a", "a") == "tie"      # position bias: each order picked "A"


def test_run_scores_systems_against_base(tmp_path):
    good = "# T\nZipline delivers blood to hospitals in Rwanda every day.\n"
    bad = "# T\nZipline sells pizza to students in Kenya every night.\n"
    (tmp_path / "eval" / "outputs").mkdir(parents=True)
    write_jsonl(tmp_path / "eval" / "outputs" / "base.jsonl", [{"episode_id": "e", "article": bad}])
    write_jsonl(tmp_path / "eval" / "outputs" / "sft.jsonl", [{"episode_id": "e", "article": good}])

    class Judge:  # prefers whichever article mentions Rwanda
        def count(self, t):
            return len(t.split())

        def chat(self, convs, max_tokens, temperature=None):
            out = []
            for c in convs:
                a = c[1]["content"].split("Article A:")[1].split("Article B:")[0]
                out.append("Winner: A" if "Rwanda" in a else "Winner: B")
            return out

    summary = judge.run(tmp_path / "eval", {"e": CHUNKS}, Judge(), tmp_path / "out")
    assert summary["sft"] == {"win": 1.0, "tie": 0.0, "loss": 0.0, "episodes": 1}
    assert (tmp_path / "out" / "judge_details.jsonl").is_file()
