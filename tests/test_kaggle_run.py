"""kaggle_run.py against a fake kaggle CLI: guard, smoke-first rule, quota log, wait."""
import csv
import json
import subprocess
from datetime import timedelta

import pytest

import build_notebook as nb
import kaggle_run as kr

USER = "tester"


class FakeKaggle:
    def __init__(self):
        self.status = {}          # ref -> status string, missing = 404
        self.pushes = []
        self.status_script = {}   # ref -> list of statuses returned in order

    def __call__(self, args, timeout=120):
        if args[:2] == ["kernels", "status"]:
            ref = args[2]
            script = self.status_script.get(ref)
            status = script.pop(0) if script else self.status.get(ref)
            if status is None:
                return subprocess.CompletedProcess(
                    args, 1, f"Cannot access kernel '{ref}' (Permission 'kernels.get' was denied). ...", "")
            return subprocess.CompletedProcess(args, 0, f'{ref} has status "KernelWorkerStatus.{status.upper()}"\n', "")
        if args[:2] == ["kernels", "push"]:
            meta = json.loads((kr.Path(args[3]) / "kernel-metadata.json").read_text())
            self.pushes.append(meta)
            self.status[meta["id"]] = "queued"
            return subprocess.CompletedProcess(
                args, 0, f"Kernel version 3 successfully pushed.  Please check progress at https://k/{meta['id']}\n", "")
        if args[:2] == ["datasets", "status"]:
            return subprocess.CompletedProcess(args, 0, "ready\n", "")
        if args[:2] == ["kernels", "output"]:
            return subprocess.CompletedProcess(args, 0, "", "")
        if args[:2] == ["kernels", "logs"]:
            return subprocess.CompletedProcess(args, 0, json.dumps([{"data": "line1\n"}, {"data": "boom\n"}]), "")
        return subprocess.CompletedProcess(args, 0, "", "")


@pytest.fixture
def fake(tmp_path, monkeypatch):
    monkeypatch.setenv("KAGGLE_USERNAME", USER)
    kr.username.cache_clear()
    monkeypatch.setattr(kr, "RUNS", tmp_path / "runs")
    monkeypatch.setattr(kr, "STATE", tmp_path / "runs" / "state.json")
    monkeypatch.setattr(kr, "QUOTA_LOG", tmp_path / "runs" / "quota_log.csv")
    real_build = nb.build
    monkeypatch.setattr(kr.nb, "build", lambda *a, **k: real_build(*a, **k, out_root=tmp_path / "build"))
    monkeypatch.setattr(kr.time, "sleep", lambda s: None)
    cli = FakeKaggle()
    monkeypatch.setattr(kr, "run_cli", cli)
    yield cli
    kr.username.cache_clear()


def push(*argv):
    return kr.main(["push", *argv])


def test_parse_status_formats():
    assert kr.parse_status('u/k has status "KernelWorkerStatus.RUNNING"') == ("running", None)
    assert kr.parse_status('u/k has status "complete"') == ("complete", None)
    assert kr.parse_status('u/k has status "KernelWorkerStatus.ERROR"\nFailure message: "OOM"') == ("error", "OOM")


def test_parse_duration_and_log_flattening():
    assert [kr.parse_duration(x) for x in ("90", "90s", "30m", "4h")] == [90, 90, 1800, 14400]
    assert kr.flatten_log(json.dumps([{"data": "a\n"}, {"data": "b\n"}])) == "a\nb\n"
    assert kr.flatten_log("plain text") == "plain text"


def test_guard_refuses_duplicate_session_unless_forced(fake):
    fake.status[f"{USER}/pod2post-00-smoke"] = "running"
    assert push("00_smoke") == 1 and fake.pushes == []
    assert push("00_smoke", "--force") == 0 and len(fake.pushes) == 1


def test_guard_requires_complete_upstream(fake):
    assert push("04_dpo") == 1                        # 03_sft-smoke never ran
    fake.status[f"{USER}/pod2post-03-sft-smoke"] = "complete"
    assert push("04_dpo") == 0
    assert fake.pushes[-1]["kernel_sources"] == [f"{USER}/pod2post-03-sft-smoke"]


def test_full_run_needs_smoke_of_same_code(fake):
    fake.status[f"{USER}/pod2post-02-teacher-gen"] = "complete"
    assert push("03_sft", "--full") == 1              # no smoke run recorded
    fake.status[f"{USER}/pod2post-02-teacher-gen-smoke"] = "complete"
    assert push("03_sft") == 0
    fake.status[f"{USER}/pod2post-03-sft-smoke"] = "complete"
    assert kr.main(["status", "03_sft"]) == 0         # records the smoke run as complete
    assert push("03_sft", "--full") == 0
    assert fake.pushes[-1]["id"] == f"{USER}/pod2post-03-sft"


def test_full_run_refused_while_another_run_is_active(fake):
    fake.status[f"{USER}/pod2post-02-teacher-gen-smoke"] = "complete"
    assert push("03_sft") == 0                        # state: 03_sft-smoke queued
    fake.status[f"{USER}/pod2post-03-sft-smoke"] = "complete"
    kr.main(["status", "03_sft"])
    assert push("01_transcribe") == 0                 # another smoke run, still queued
    fake.status[f"{USER}/pod2post-02-teacher-gen"] = "complete"
    assert push("03_sft", "--full") == 1
    assert "pod2post-03-sft" not in [p["id"].split("/")[1] for p in fake.pushes]


def test_quota_row_logged_once_with_kernel_runtime(fake):
    assert push("00_smoke") == 0
    state = kr.load_state()
    run = state["runs"]["pod2post-00-smoke"]
    dest = kr.pull_dir(run)
    dest.mkdir(parents=True)
    (dest / "run_manifest.json").write_text(json.dumps({"code_hash": run["code_hash"], "runtime_s": 360}))
    fake.status[f"{USER}/pod2post-00-smoke"] = "complete"
    assert kr.main(["pull", "00_smoke"]) == 0
    assert kr.main(["status", "00_smoke"]) == 0       # second sighting must not add a row
    rows = list(csv.DictReader(kr.QUOTA_LOG.open()))
    assert len(rows) == 1
    assert rows[0]["status"] == "complete" and rows[0]["kernel_runtime_hours"] == "0.100"
    assert dest.name == "00_smoke-v3"


def test_weekly_warning(fake, capsys):
    now = kr.utcnow()
    for days_ago, hours in ((1, 15.0), (2, 6.0), (9, 50.0)):
        kr.append_quota_row({"kernel": "03_sft", "slug": "s", "mode": "full", "version": 1, "code_hash": "h",
                             "pushed_at": "", "finished_at": (now - timedelta(days=days_ago)).isoformat(),
                             "wall_hours": "", "kernel_runtime_hours": f"{hours}", "status": "complete"})
    assert kr.quota_totals(now) == (21.0, 71.0)
    kr.warn_quota()
    assert "WARNING" in capsys.readouterr().out


def test_wait_ignores_stale_status_right_after_push(fake, capsys):
    assert push("00_smoke") == 0
    ref = f"{USER}/pod2post-00-smoke"
    fake.status_script[ref] = ["complete", "queued", "running", "complete"]  # first is the previous version
    assert kr.main(["wait", "00_smoke", "--interval", "0"]) == 0
    out = capsys.readouterr().out
    assert "running" in out
    assert kr.load_state()["runs"]["pod2post-00-smoke"]["status"] == "complete"


def test_error_prints_log_tail(fake, capsys):
    fake.status[f"{USER}/pod2post-00-smoke"] = "error"
    assert kr.main(["status", "00_smoke"]) == 1
    assert "boom" in capsys.readouterr().out


def test_dataset_status_treats_own_403_as_absent(fake, monkeypatch):
    def cli(args, timeout=120):
        if args[:2] == ["datasets", "status"]:
            if args[2] == f"{USER}/pod2post-data":
                return subprocess.CompletedProcess(args, 0, "ready\n", "")
            return subprocess.CompletedProcess(args, 0, "403 Client Error: Forbidden for url: ...", "")
        return fake(args, timeout)
    monkeypatch.setattr(kr, "run_cli", cli)
    assert kr.dataset_status(f"{USER}/pod2post-data") == "ready"
    assert kr.dataset_status(f"{USER}/missing") == "absent"


def test_status_survives_a_network_blip(fake, monkeypatch):
    calls = {"n": 0}

    def flaky(args, timeout=120):
        if args[:2] == ["kernels", "status"]:
            calls["n"] += 1
            if calls["n"] <= 2:
                return subprocess.CompletedProcess(args, 1, "", "Max retries exceeded (NameResolutionError)")
        return fake(args, timeout)

    fake.status[f"{USER}/pod2post-00-smoke"] = "running"
    monkeypatch.setattr(kr, "run_cli", flaky)
    assert kr.kernel_status(f"{USER}/pod2post-00-smoke") == ("running", None) and calls["n"] == 3


def test_full_stage_runs_its_smoke_first_when_code_changed(fake, monkeypatch):
    monkeypatch.setattr(kr, "PUSH_GRACE_S", 0)
    real_push = kr.cmd_push

    def push_and_finish(args):  # the fake run completes as soon as it is pushed
        rc = real_push(args)
        fake.status[kr.kernel_ref(args.kernel, args.full)] = "complete"
        return rc

    monkeypatch.setattr(kr, "cmd_push", push_and_finish)
    args = kr.argparse.Namespace(full=True, timeout=60, interval=0)
    assert kr.run_stage("01_transcribe", args) is not None
    assert [p["id"] for p in fake.pushes] == [f"{USER}/pod2post-01-transcribe-smoke", f"{USER}/pod2post-01-transcribe"]
    assert kr.run_stage("01_transcribe", args) is not None          # smoke is now current: no second smoke
    assert [p["id"] for p in fake.pushes][-1] == f"{USER}/pod2post-01-transcribe"


def test_dataset_status_survives_a_network_blip(fake, monkeypatch):
    calls = {"n": 0}

    def flaky(args, timeout=120):
        if args[:2] == ["datasets", "status"]:
            calls["n"] += 1
            if calls["n"] == 1:
                return subprocess.CompletedProcess(args, 0, "('Connection aborted.', RemoteDisconnected('x'))", "")
            return subprocess.CompletedProcess(args, 0, "ready\n", "")
        return fake(args, timeout)

    monkeypatch.setattr(kr, "run_cli", flaky)
    assert kr.dataset_status(f"{USER}/pod2post-audio") == "ready" and calls["n"] == 2
