#!/usr/bin/env python3
"""Drive the pod2post Kaggle kernels from a terminal (spec §7.3).

  python scripts/kaggle_run.py check               credentials, network, quota
  python scripts/kaggle_run.py push 03_sft         build notebook, pre-flight guard, push (SMOKE run)
  python scripts/kaggle_run.py push 03_sft --full  full run; needs a completed smoke run of the same code
  python scripts/kaggle_run.py status 03_sft       queued / running / complete / error
  python scripts/kaggle_run.py wait 03_sft         poll every 60s until done (timeout 4h)
  python scripts/kaggle_run.py pull 03_sft         outputs -> runs/<date>/03_sft[-smoke]-v<N>/
  python scripts/kaggle_run.py logs 03_sft         last 50 log lines
  python scripts/kaggle_run.py pipeline            01 -> 05 in order, stop on first error
  python scripts/kaggle_run.py quota               local quota log + Kaggle's weekly counter
  python scripts/kaggle_run.py dataset [audio]     upload data/processed/ (or data/raw/audio/) as a private dataset

Every command targets the SMOKE kernel unless --full is given.
Credentials are read only by the kaggle CLI itself (KAGGLE_API_TOKEN,
~/.kaggle/access_token, `kaggle auth login`, or legacy kaggle.json /
KAGGLE_USERNAME + KAGGLE_KEY). This script never opens, prints or writes them.
"""
from __future__ import annotations

import argparse
import csv
import functools
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_notebook as nb  # noqa: E402

REPO = nb.REPO
RUNS = REPO / "runs"
STATE = RUNS / "state.json"
QUOTA_LOG = RUNS / "quota_log.csv"
QUOTA_FIELDS = ["kernel", "slug", "mode", "version", "code_hash", "pushed_at", "finished_at",
                "wall_hours", "kernel_runtime_hours", "status"]
PIPELINE = ["01_transcribe", "02_teacher_gen", "03_sft", "04_dpo", "05_eval"]
DATASETS = {  # name -> (Kaggle slug, local folder, files; None = every file in the folder)
    "data": ("pod2post-data", "data/processed", None),
    "data-smoke": ("pod2post-data-smoke", "data/smoke", None),  # smoke runs never touch the real dataset
    "audio": ("pod2post-audio", "data/raw/audio", None),
}
BASE_DATA_FILES = ["episodes.jsonl", "chunks.jsonl", "test_ids.sha256"]
LOCAL_PY = str(REPO / ".venv" / "bin" / "python") if (REPO / ".venv" / "bin" / "python").is_file() else sys.executable
MAX_RESUMES = 3
PULL_RETRIES = 4
ABSENT_RETRIES = 5  # the status API sometimes answers "denied" for a kernel that exists
ACTIVE = {"queued", "running", "new_script", "cancel_requested"}
TERMINAL = {"complete", "error", "cancel_acknowledged"}
WEEKLY_WARN_HOURS = 20.0
RUN_TIMEOUT_S = {"smoke": 20 * 60, "full": 3 * 3600}  # spec: smoke < 10 min, every job < 3 h
# Kernels that install vLLM spend ~8-12 min on install + model download + engine start before any work.
SMOKE_TIMEOUT_OVERRIDES = {"02_teacher_gen": 50 * 60, "04_dpo": 50 * 60, "05_eval": 50 * 60}
PUSH_GRACE_S = 180  # `kernels status` has no version argument; right after a push it may still report the old run
HOSTS = {
    "api.kaggle.com": "Kaggle API (kaggle CLI 2.x)",
    "www.kaggle.com": "Kaggle web / legacy API",
    "kaggle.com": "Kaggle web",
    "storage.googleapis.com": "file downloads/uploads (presumed: signed URLs)",
}
SECRET_FILES = ["kaggle.json", "access_token", "credentials.json", ".env"]

STATUS_RE = re.compile(r'has status "([^"]+)"')
FAILURE_RE = re.compile(r'Failure message: "(.*)"', re.S)
PUSH_OK_RE = re.compile(r"Kernel version (\d+) successfully pushed")
INVALID_SOURCES_RE = re.compile(r"not valid (?:dataset|kernel|competition) sources.*", re.I)


# --- small helpers ---------------------------------------------------------

def utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def copy_verified(src: Path, dst: Path) -> None:
    """Copy and confirm the bytes match: a local disk fault once zeroed a 16 KB block of a copy."""
    want = sha256_file(src)
    for _ in range(2):
        shutil.copy2(src, dst)
        if sha256_file(dst) == want:
            return
    die(f"copy of {src} keeps differing from the source; the disk may be failing (run Disk Utility > First Aid)")


def die(msg: str) -> None:
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(1)


def parse_duration(text: str) -> int:
    m = re.fullmatch(r"(\d+(?:\.\d+)?)([smh]?)", text.strip())
    if not m:
        raise argparse.ArgumentTypeError(f"bad duration {text!r} (use e.g. 90s, 30m, 4h)")
    return int(float(m[1]) * {"": 1, "s": 1, "m": 60, "h": 3600}[m[2]])


def rel(path: Path) -> str:
    return path.relative_to(REPO).as_posix() if path.is_relative_to(REPO) else str(path)


def tail(text: str, n: int = 50) -> str:
    return "\n".join(text.rstrip().splitlines()[-n:])


# --- kaggle CLI ------------------------------------------------------------

def kaggle_bin() -> str:
    path = os.environ.get("KAGGLE_CLI") or shutil.which("kaggle")
    if not path:
        die("kaggle CLI not found on PATH; run `pip install -r requirements.txt`")
    return path


def run_cli(args: list[str], timeout: int = 120) -> subprocess.CompletedProcess:
    try:
        return subprocess.run([kaggle_bin(), *args], capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return subprocess.CompletedProcess(args, 124, "", f"kaggle {' '.join(args)} timed out after {timeout}s")


@functools.cache
def username() -> str:
    name = os.environ.get("KAGGLE_USERNAME")
    if not name:
        out = run_cli(["config", "view"])
        m = re.search(r"^- username: (\S+)", out.stdout, re.M)
        name = m[1] if m and m[1] != "None" else None
    if not name:
        die("could not determine your Kaggle username. Run `python scripts/kaggle_run.py check`, "
            "or export KAGGLE_USERNAME=<your username> (it is not a secret).")
    return name


def kernel_ref(kernel: str, full: bool) -> str:
    if not (nb.KAGGLE_DIR / kernel).is_dir():
        die(f"unknown kernel {kernel!r}; choose from {sorted(p.name for p in nb.KAGGLE_DIR.iterdir() if p.is_dir())}")
    if kernel == "00_smoke" and full:
        die("00_smoke only has a smoke mode")
    return f"{username()}/{nb.mode_slug(nb.template_slug(kernel), smoke=not full)}"


def parse_status(text: str) -> tuple[str, str | None]:
    """'u/k has status "KernelWorkerStatus.RUNNING"' -> ('running', None)."""
    m = STATUS_RE.search(text)
    if not m:
        raise ValueError(f"unrecognised status output: {text!r}")
    failure = FAILURE_RE.search(text)
    return m[1].rsplit(".", 1)[-1].lower(), (failure[1] if failure else None)


def kernel_status(ref: str) -> tuple[str, str | None]:
    out = run_cli(["kernels", "status", ref], timeout=60)
    text = out.stdout + out.stderr
    if STATUS_RE.search(text):
        return parse_status(text)
    if re.search(r"\b404\b|not found", text, re.I):
        return "absent", None
    # CLI 2.x answers "Permission 'kernels.get' was denied" for a kernel that doesn't exist yet;
    # under our own username that can only mean it has never been pushed.
    if "kernels.get' was denied" in text and ref.split("/", 1)[0] == username():
        return "absent", None
    die(f"`kaggle kernels status {ref}` failed:\n{text.strip()}")


def dataset_status(ref: str) -> str:
    """'ready' / 'pending' / ... or 'absent'. A missing dataset comes back as 403 (with exit code 0)."""
    out = run_cli(["datasets", "status", ref], timeout=60)
    text = (out.stdout + out.stderr).strip()
    if "403" in text or "404" in text:
        if ref.split("/", 1)[0] == username():
            return "absent"
        die(f"cannot access dataset {ref}: {text}")
    return text.splitlines()[-1].strip().lower() if text else "unknown"


def flatten_log(text: str) -> str:
    """Kaggle logs come as a JSON list of {stream_name, time, data}; fall back to raw text."""
    try:
        entries = json.loads(text)
    except ValueError:
        return text
    if isinstance(entries, list) and all(isinstance(e, dict) for e in entries):
        return "".join(str(e.get("data", "")) for e in entries)
    return text


def print_error_log(ref: str, lines: int = 50) -> None:
    out = run_cli(["kernels", "logs", ref], timeout=180)
    print(f"--- last {lines} log lines of {ref} ---")
    print(tail(flatten_log(out.stdout or out.stderr), lines))


# --- state + quota log (committed, so cloud sessions can pick up) ----------

def load_state() -> dict:
    if STATE.is_file():
        return json.loads(STATE.read_text(encoding="utf-8"))
    return {"phases": {}, "runs": {}, "artifacts": {}}


def save_state(state: dict) -> None:
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_quota_log() -> list[dict]:
    if not QUOTA_LOG.is_file():
        return []
    with QUOTA_LOG.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def append_quota_row(row: dict) -> None:
    new = not QUOTA_LOG.is_file()
    QUOTA_LOG.parent.mkdir(parents=True, exist_ok=True)
    with QUOTA_LOG.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=QUOTA_FIELDS)
        if new:
            w.writeheader()
        w.writerow(row)


def row_hours(row: dict) -> float:
    """In-kernel runtime when the manifest was readable, else push-to-finish wall time (an upper bound)."""
    return float(row.get("kernel_runtime_hours") or row.get("wall_hours") or 0)


def quota_totals(now: datetime | None = None) -> tuple[float, float]:
    now = now or utcnow()
    rows = read_quota_log()
    week = sum(row_hours(r) for r in rows
               if r.get("finished_at") and now - datetime.fromisoformat(r["finished_at"]) <= timedelta(days=7))
    return week, sum(row_hours(r) for r in rows)


def warn_quota() -> None:
    week, total = quota_totals()
    print(f"GPU hours logged: {week:.2f} h in the last 7 days, {total:.2f} h project total (budget 20 h)")
    if week > WEEKLY_WARN_HOURS:
        print(f"WARNING: more than {WEEKLY_WARN_HOURS:.0f} GPU-hours used this week; check `kaggle quota`.")


def pull_dir(run: dict) -> Path:
    """runs/<push date>/<kernel>[-smoke]-v<version>/, so repeated runs never mix files."""
    name = run["kernel"] if run["mode"] == "full" or run["kernel"].endswith("smoke") else f"{run['kernel']}-smoke"
    return RUNS / run["pushed_at"][:10] / f"{name}-v{run.get('version', 'unknown')}"


def read_manifest(ref: str, dest: Path, fetch: bool = True) -> dict | None:
    path = dest / "run_manifest.json"
    if fetch:
        dest.mkdir(parents=True, exist_ok=True)
        run_cli(["kernels", "output", ref, "-p", str(dest), "--file-pattern", r"run_manifest\.json$", "-o", "-q"],
                timeout=300)
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def record_status(state: dict, ref: str, status: str, failure: str | None, pulled: bool = False) -> None:
    """Store the latest status; the first time a run is seen finished, append it to the quota log."""
    slug = ref.split("/", 1)[1]
    run = state["runs"].get(slug)
    if run is None:
        return
    run["status"] = status
    if failure:
        run["failure_message"] = failure
    if status in TERMINAL and not run.get("logged"):
        now = utcnow()
        manifest = read_manifest(ref, pull_dir(run), fetch=not pulled)
        runtime = manifest.get("runtime_s") if manifest and manifest.get("code_hash") == run["code_hash"] else None
        run["finished_at"] = now.isoformat()
        append_quota_row({
            "kernel": run["kernel"], "slug": slug, "mode": run["mode"], "version": run["version"],
            "code_hash": run["code_hash"], "pushed_at": run["pushed_at"], "finished_at": run["finished_at"],
            "wall_hours": f"{(now - datetime.fromisoformat(run['pushed_at'])).total_seconds() / 3600:.3f}",
            "kernel_runtime_hours": f"{runtime / 3600:.3f}" if runtime else "",
            "status": status,
        })
        run["logged"] = True
    save_state(state)


# --- commands --------------------------------------------------------------

def credential_sources() -> list[Path | str]:
    """Where the kaggle CLI could find credentials. Presence only; nothing is opened."""
    cfg = Path(os.environ.get("KAGGLE_CONFIG_DIR") or Path.home() / ".kaggle")
    found: list[Path | str] = []
    if os.environ.get("KAGGLE_API_TOKEN"):
        found.append("KAGGLE_API_TOKEN env var")
    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        found.append("KAGGLE_USERNAME + KAGGLE_KEY env vars (legacy key)")
    found += [p for p in (cfg / "access_token", cfg / "credentials.json", cfg / "kaggle.json") if p.is_file()]
    return found


def check_secret_hygiene() -> list[str]:
    problems = []
    tracked = subprocess.run(["git", "ls-files"], cwd=REPO, capture_output=True, text=True).stdout.split()
    problems += [f"secret-looking file is tracked by git: {t}" for t in tracked if Path(t).name in SECRET_FILES]
    for name in SECRET_FILES:
        for p in REPO.rglob(name):
            ignored = subprocess.run(["git", "check-ignore", "-q", str(p)], cwd=REPO).returncode == 0
            if not ignored:
                problems.append(f"{p.relative_to(REPO)} exists in the repo and is NOT git-ignored")
    return problems


def check_host(host: str) -> str:
    try:
        urllib.request.urlopen(urllib.request.Request(f"https://{host}/", method="HEAD"), timeout=10)
        return "reachable"
    except urllib.error.HTTPError as e:
        blocked = e.code in (403, 407) and "x-deny-reason" in {k.lower() for k in e.headers.keys()}
        return f"BLOCKED by proxy (HTTP {e.code})" if blocked else f"reachable (HTTP {e.code})"
    except (urllib.error.URLError, OSError) as e:
        return f"UNREACHABLE ({getattr(e, 'reason', e)})"


def cmd_check(args) -> int:
    ok = True
    version = run_cli(["--version"]).stdout.strip()
    print(f"✓ kaggle CLI: {kaggle_bin()} ({version})")

    sources = credential_sources()
    if sources:
        for s in sources:
            print(f"✓ credentials source present: {s}")
            if isinstance(s, Path) and s.stat().st_mode & 0o077:
                print(f"  ✗ {s} is readable by other users; run: chmod 600 '{s}'")
                ok = False
    else:
        print("? no token file or env var found (fine if you used `kaggle auth login`; verified below)")

    for p in check_secret_hygiene():
        print(f"✗ {p}")
        ok = False

    for host, why in HOSTS.items():
        result = check_host(host)
        print(f"{'✓' if result.startswith('reachable') else '✗'} {host:24s} {result:28s} {why}")

    out = run_cli(["kernels", "list", "--mine"], timeout=90)
    text = (out.stdout + out.stderr).strip()
    if out.returncode == 0 and "Authentication required" not in text:
        print("✓ `kaggle kernels list --mine` works")
        print(f"✓ Kaggle username: {username()}")
        quota = run_cli(["quota"], timeout=60)
        print("Kaggle weekly accelerator quota:\n" + (quota.stdout or quota.stderr).rstrip())
    else:
        ok = False
        print("✗ `kaggle kernels list --mine` failed:\n" + tail(text, 15))
        if re.search(r"Authentication required|401|Unauthenticated|Unauthorized", text, re.I):
            print("  -> Credentials missing or invalid. Run `kaggle auth login`, or save a token from "
                  "kaggle.com/settings/api to ~/.kaggle/access_token (chmod 600), or export KAGGLE_API_TOKEN.")
        else:
            print("  -> Looks like a network block. In a Claude Code cloud environment, allow outbound access to "
                  + ", ".join(HOSTS) + " (custom network allowlist).")
    warn_quota()
    return 0 if ok else 1


def preflight(kernel: str, full: bool, state: dict, meta: dict, digest: str) -> list[str]:
    """Reasons not to push (each one overridable with --force)."""
    problems = []
    own = meta["id"]
    own_status, _ = kernel_status(own)
    if own_status in ACTIVE:
        problems.append(f"{own} is already {own_status}; pushing again would start a duplicate GPU session")
    for src in meta["kernel_sources"]:
        if src == own:
            if own_status == "absent":
                problems.append(f"--resume given but {own} has never run")
            continue
        status, _ = kernel_status(src)
        if status != "complete":
            problems.append(f"upstream {src} is {status}, not complete")
    for src in meta.get("dataset_sources", []):
        status = dataset_status(src)
        if status != "ready":
            problems.append(f"dataset {src} is {status}, not ready")
    if full:
        smoke = state["runs"].get(nb.mode_slug(nb.template_slug(kernel), smoke=True), {})
        if smoke.get("status") != "complete" or smoke.get("code_hash") != digest:
            problems.append(f"no completed SMOKE run of this exact code (hash {digest}); "
                            f"run `kaggle_run.py push {kernel}` + `wait` first")
        for slug, run in state["runs"].items():
            if slug != own.split("/", 1)[1] and run.get("status") not in TERMINAL:
                status, _ = kernel_status(f"{username()}/{slug}")
                if status in ACTIVE:
                    problems.append(f"{slug} is {status}; never start a full run while another run is active")
    return problems


def cmd_push(args) -> int:
    ref = kernel_ref(args.kernel, args.full)
    mode = "full" if args.full else "smoke"
    state = load_state()
    out_dir, meta, digest = nb.build(args.kernel, username(), smoke=not args.full, resume=args.resume)
    print(f"built {meta['id']} ({mode}, code hash {digest}) -> {rel(out_dir)}")

    problems = preflight(args.kernel, args.full, state, meta, digest)
    for p in problems:
        print(f"  ✗ {p}")
    if problems and not args.force:
        print("Refusing to push. Fix the above or pass --force.")
        return 1
    warn_quota()

    timeout = args.run_timeout or (SMOKE_TIMEOUT_OVERRIDES.get(args.kernel) if mode == "smoke" else None) \
        or RUN_TIMEOUT_S[mode]
    out = run_cli(["kernels", "push", "-p", str(out_dir), "-t", str(timeout)], timeout=600)
    text = (out.stdout + out.stderr).strip()
    print(text)
    m = PUSH_OK_RE.search(text)
    if out.returncode != 0 or not m:
        print(f"Push of {ref} failed.")
        return 1
    state["runs"][ref.split("/", 1)[1]] = {
        "kernel": args.kernel, "mode": mode, "version": int(m[1]), "code_hash": digest,
        "pushed_at": utcnow().isoformat(), "status": "queued", "logged": False, "resume": args.resume,
    }
    save_state(state)
    if INVALID_SOURCES_RE.search(text):
        print("ERROR: Kaggle rejected some input sources (see above); this run will not see its inputs.")
        return 1
    return 0


def cmd_status(args) -> int:
    ref = kernel_ref(args.kernel, args.full)
    status, failure = kernel_status(ref)
    print(f"{ref}: {status}" + (f"  (failure: {failure})" if failure else ""))
    record_status(load_state(), ref, status, failure)
    if status == "error":
        print_error_log(ref)
        return 1
    return 0


def cmd_wait(args) -> int:
    ref = kernel_ref(args.kernel, args.full)
    run = load_state()["runs"].get(ref.split("/", 1)[1], {})
    pushed = datetime.fromisoformat(run["pushed_at"]) if "pushed_at" in run else None
    deadline = time.time() + args.timeout
    last, seen_active, absent_seen = None, False, 0
    while True:
        status, failure = kernel_status(ref)
        if status == "absent" and pushed is not None and absent_seen < ABSENT_RETRIES:
            absent_seen += 1  # we pushed this run, so "absent" is a transient API answer
            time.sleep(args.interval)
            continue
        seen_active |= status in ACTIVE
        if status != last:
            print(f"[{utcnow():%H:%M:%S}Z] {ref}: {status}", flush=True)
            last = status
        stale = (status in TERMINAL and not seen_active and pushed is not None
                 and (utcnow() - pushed).total_seconds() < PUSH_GRACE_S)
        if status == "absent" or (status in TERMINAL and not stale):
            break
        if time.time() > deadline:
            print(f"Timed out after {args.timeout}s; the run is still {status}. Check again with `status`.")
            return 2
        time.sleep(15 if stale else args.interval)
    record_status(load_state(), ref, status, failure)
    if status == "complete":
        return 0
    if status != "absent":
        print_error_log(ref)
    return 1


def cmd_pull(args) -> int:
    ref = kernel_ref(args.kernel, args.full)
    slug = ref.split("/", 1)[1]
    status, failure = kernel_status(ref)
    if status in ACTIVE:
        print(f"{ref} is still {status}; run `wait` first (pulling now would fetch an older version's output).")
        return 1
    state = load_state()
    run = state["runs"].get(slug) or {"kernel": args.kernel, "mode": "full" if args.full else "smoke",
                                      "pushed_at": utcnow().isoformat()}
    dest = pull_dir(run)
    dest.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, PULL_RETRIES + 1):  # downloads sometimes die with "connection reset"
        out = run_cli(["kernels", "output", ref, "-p", str(dest), "-o"], timeout=1800)
        text = (out.stdout + out.stderr).strip()
        if out.returncode == 0 and not re.search(r"Connection (broken|reset)|Max retries", text):
            break
        print(f"pull attempt {attempt}/{PULL_RETRIES} failed: {tail(text, 3)}", flush=True)
        if attempt == PULL_RETRIES:
            return 1
        time.sleep(30 * attempt)
    files = sorted(p.relative_to(dest).as_posix() for p in dest.rglob("*") if p.is_file())
    print(f"pulled {len(files)} files -> {rel(dest)}")
    for f in files[:40]:
        print(f"  {f}")
    if slug in state["runs"]:
        state["runs"][slug]["pulled_to"] = rel(dest)
        record_status(state, ref, status, failure, pulled=True)
    return 0


def cmd_logs(args) -> int:
    print_error_log(kernel_ref(args.kernel, args.full), args.lines)
    return 0


def local(*module_args: str, capture: Path | None = None) -> int:
    """Run a repo module locally (in .venv when present), e.g. local("src.data.filter", dir)."""
    print(f"$ python -m {' '.join(module_args)}", flush=True)
    out = subprocess.run([LOCAL_PY, "-m", *module_args], cwd=REPO, capture_output=capture is not None, text=True)
    if capture is not None:
        capture.write_text(out.stdout, encoding="utf-8")
        print(out.stdout + out.stderr, flush=True)
    return out.returncode


def pulled_dir(kernel: str, full: bool) -> Path:
    return REPO / load_state()["runs"][kernel_ref(kernel, full).split("/", 1)[1]]["pulled_to"]


def needs_resume(kernel: str, out: Path) -> bool:
    if kernel == "02_teacher_gen":
        stats = out / "teacher" / "teacher_stats.json"
        return stats.is_file() and bool(json.loads(stats.read_text()).get("stopped_early"))
    key = {"03_sft": "sft", "04_dpo": "dpo"}.get(kernel)
    manifest = out / "run_manifest.json"
    return bool(key and manifest.is_file() and json.loads(manifest.read_text()).get(key, {}).get("stopped_by_time_budget"))


def run_stage(kernel: str, args) -> Path | None:
    """push -> wait -> pull, then resume with --resume while the run stopped at its time budget."""
    resume = False
    for attempt in range(MAX_RESUMES + 1):
        step = argparse.Namespace(kernel=kernel, full=args.full, force=False, resume=resume, run_timeout=None,
                                  timeout=args.timeout, interval=args.interval)
        for fn in (cmd_push, cmd_wait, cmd_pull):
            if fn(step) != 0:
                print(f"Pipeline stopped at {kernel} ({fn.__name__[4:]}).")
                return None
        out = pulled_dir(kernel, args.full)
        if not needs_resume(kernel, out):
            return out
        print(f"{kernel} stopped at its time budget; resuming ({attempt + 1}/{MAX_RESUMES}).", flush=True)
        resume = True
    print(f"{kernel} is still unfinished after {MAX_RESUMES} resumes.")
    return None


def wait_dataset_ready(name: str, timeout_s: int = 1800) -> bool:
    ref = f"{username()}/{DATASETS[name][0]}"
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        status = dataset_status(ref)
        if status == "ready":
            return True
        print(f"dataset {ref}: {status}", flush=True)
        time.sleep(30)
    return False


def seed_smoke_dataset() -> bool:
    """Smoke kernels read pod2post-data-smoke from 02 on; give it the base files before 02 runs."""
    out = REPO / "data" / "smoke"
    out.mkdir(parents=True, exist_ok=True)
    for f in BASE_DATA_FILES:
        copy_verified(REPO / "data" / "processed" / f, out / f)
    ns = argparse.Namespace(name="data-smoke", message="base files", force=False)
    return cmd_dataset(ns) == 0 and wait_dataset_ready("data-smoke")


def build_training_data(teacher_out: Path, full: bool) -> bool:
    """Between 02 and 03: filter teacher outputs, build SFT + DPO data, upload the dataset."""
    tdir = teacher_out / "teacher"
    if local("src.data.filter", str(tdir)) != 0:
        return False
    report = json.loads((tdir / "filter_report.json").read_text())
    if full:
        cfg = __import__("yaml").safe_load((REPO / "configs" / "teacher.yaml").read_text())["targets"]
        kept_a, kept_b = report["stage_a"]["kept"], report["stage_b"]["kept"]
        if kept_a < 0.7 * cfg["stage_a"] or kept_b < 0.7 * cfg["stage_b"]:  # Phase 3 acceptance (spec §9)
            print(f"Phase 3 acceptance failed: {kept_a} Stage A / {kept_b} Stage B survived filtering "
                  f"(need >= 70% of {cfg['stage_a']} / {cfg['stage_b']}). See {rel(tdir / 'filter_report.json')}.")
            return False
        out, dataset = REPO / "data" / "processed", "data"
    else:
        out, dataset = REPO / "data" / "smoke", "data-smoke"
        out.mkdir(parents=True, exist_ok=True)
        for f in BASE_DATA_FILES:
            copy_verified(REPO / "data" / "processed" / f, out / f)
        kept = tdir / "filtered_stage_b.jsonl"
        if len(kept.read_text().splitlines()) < 3:
            # Smoke only tests plumbing: top up with parseable rejected rows so 03/04 have data.
            from src.eval.seo import parse_output  # noqa: E402 (repo import, local only)
            extra = [line for line in (tdir / "rejected_stage_b.jsonl").read_text().splitlines()
                     if all(parse_output(json.loads(line)["output"], repair=True))]
            kept.write_text(kept.read_text() + "".join(line + "\n" for line in extra))
    for module in ("src.data.build_sft", "src.data.build_dpo"):
        if local(module, str(tdir), str(out)) != 0:
            return False
    ns = argparse.Namespace(name=dataset, message=f"from {teacher_out.name}", force=False)
    return cmd_dataset(ns) == 0 and wait_dataset_ready(dataset)


def cmd_pipeline(args) -> int:
    """The whole project: 01 -> 05 with the local steps in between (spec §7.3, plan.md)."""
    sys.path.insert(0, str(REPO))
    stages = PIPELINE[PIPELINE.index(args.start):] if args.start else PIPELINE
    mode = "full" if args.full else "smoke"
    todo = [k for k in stages if k not in args.skip]
    if not args.full and any(k != "01_transcribe" for k in todo) and not seed_smoke_dataset():
        print("Pipeline stopped: could not prepare the smoke dataset.")
        return 1
    for kernel in todo:
        print(f"\n=== {kernel} ({mode}) ===", flush=True)
        out = run_stage(kernel, args)
        if out is None:
            return 1
        if kernel == "01_transcribe" and args.full:
            wer_file = out / "transcripts" / "changelog-661.json"
            if wer_file.is_file():
                local("src.asr.wer", str(wer_file), capture=out / "wer.json")
        if kernel == "02_teacher_gen" and not build_training_data(out, args.full):
            print("Pipeline stopped after 02_teacher_gen (local data build).")
            return 1
        if kernel == "05_eval":
            target = [] if args.full else ["--out", str(out / "results.md")]
            local("src.eval.report", str(out), *target)
    print(f"\nPipeline complete ({mode}).")
    return 0


def cmd_dataset(args) -> int:
    """Create or version a private Kaggle dataset from a local folder (skipped if unchanged)."""
    slug, folder, files = DATASETS[args.name]
    src = REPO / folder
    if files is None:
        files = sorted(p.name for p in src.glob("*") if p.is_file()) if src.is_dir() else []
    missing = [f for f in files if not (src / f).is_file()]
    if missing or not files:
        die(f"missing {missing or 'files'} in {rel(src)}; build that data first")
    ref = f"{username()}/{slug}"
    stage = REPO / "build" / "dataset" / slug
    shutil.rmtree(stage, ignore_errors=True)
    stage.mkdir(parents=True)
    digest = hashlib.sha256()
    for f in files:
        copy_verified(src / f, stage / f)
        digest.update(f.encode() + (src / f).read_bytes())
    digest = digest.hexdigest()[:12]
    (stage / "dataset-metadata.json").write_text(json.dumps(
        {"title": slug, "id": ref, "licenses": [{"name": "other"}]}, indent=2), encoding="utf-8")

    state = load_state()
    known = state["artifacts"].setdefault("datasets", {}).get(args.name, {})
    status = dataset_status(ref)
    if status != "absent" and known.get("content_hash") == digest and not args.force:
        print(f"{ref} already has this content (hash {digest}); nothing to upload.")
        return 0
    cmd = (["datasets", "create", "-p", str(stage), "-t", "-q"] if status == "absent"
           else ["datasets", "version", "-p", str(stage), "-t", "-q", "-m", args.message or f"content {digest}"])
    out = run_cli(cmd, timeout=1800)
    text = (out.stdout + out.stderr).strip()
    print(text)
    if out.returncode != 0 or re.search(r"error|failed", text, re.I):
        return 1
    state["artifacts"]["datasets"][args.name] = {"id": ref, "content_hash": digest, "files": files,
                                                 "uploaded_at": utcnow().isoformat(), "private": True}
    save_state(state)
    print(f"{'created' if status == 'absent' else 'versioned'} private dataset {ref} (content {digest}); "
          "Kaggle needs a few minutes to process it (`kaggle datasets status`).")
    return 0


def cmd_quota(args) -> int:
    rows = read_quota_log()
    by_kernel: dict[str, float] = {}
    for r in rows:
        by_kernel[f"{r['kernel']} ({r['mode']})"] = by_kernel.get(f"{r['kernel']} ({r['mode']})", 0) + row_hours(r)
    for k, h in sorted(by_kernel.items()):
        print(f"  {k:28s} {h:6.2f} h")
    warn_quota()
    out = run_cli(["quota"], timeout=60)
    print("Kaggle weekly accelerator quota:\n" + (out.stdout or out.stderr).rstrip())
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def kernel_cmd(name, fn, help):
        p = sub.add_parser(name, help=help)
        p.add_argument("kernel", help="folder under kaggle/, e.g. 03_sft")
        p.add_argument("--full", action="store_true", help="target the full-run kernel (default: smoke)")
        p.set_defaults(fn=fn)
        return p

    sub.add_parser("check", help="verify CLI, credentials, network, quota").set_defaults(fn=cmd_check)
    p = kernel_cmd("push", cmd_push, "build + guard + push (starts a GPU run)")
    p.add_argument("--force", action="store_true", help="push even if the pre-flight guard objects")
    p.add_argument("--resume", action="store_true", help="attach own previous output to resume from checkpoints")
    p.add_argument("--run-timeout", type=parse_duration, help="Kaggle-side run time limit (default 20m / 3h)")
    kernel_cmd("status", cmd_status, "show the latest run's status")
    p = kernel_cmd("wait", cmd_wait, "poll until the run finishes")
    p.add_argument("--timeout", type=parse_duration, default=4 * 3600)
    p.add_argument("--interval", type=parse_duration, default=60)
    kernel_cmd("pull", cmd_pull, "download outputs to runs/<date>/<kernel>/")
    p = kernel_cmd("logs", cmd_logs, "print the tail of the latest run's log")
    p.add_argument("-n", "--lines", type=int, default=50)
    p = sub.add_parser("pipeline", help="push/wait/pull 01 -> 05, stop on first error")
    p.add_argument("--full", action="store_true")
    p.add_argument("--from", dest="start", choices=PIPELINE)
    p.add_argument("--skip", nargs="*", default=[], choices=PIPELINE)
    p.add_argument("--timeout", type=parse_duration, default=8 * 3600, help="per-stage wait (queue + run)")
    p.add_argument("--interval", type=parse_duration, default=60)
    p.set_defaults(fn=cmd_pipeline)
    sub.add_parser("quota", help="GPU hours used (local log + Kaggle counter)").set_defaults(fn=cmd_quota)
    p = sub.add_parser("dataset", help="create/version a private dataset (data: data/processed, audio: data/raw/audio)")
    p.add_argument("name", choices=sorted(DATASETS), nargs="?", default="data")
    p.add_argument("-m", "--message", help="version notes")
    p.add_argument("--force", action="store_true", help="upload even if the content is unchanged")
    p.set_defaults(fn=cmd_dataset)

    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
