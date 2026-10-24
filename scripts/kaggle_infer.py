#!/usr/bin/env python3
"""Write articles on Kaggle's GPUs with the already-trained adapters (kernel 07_infer). Nothing is trained.

  python scripts/kaggle_infer.py adapters                       upload the adapters once (private dataset)
  python scripts/kaggle_infer.py deps                           pre-install vLLM once (07_deps, CPU only, ~15 min)
  python scripts/kaggle_infer.py run --url URL [--episode E] [--title T] [--keyword K] [--model sft_dpo|sft|base]
  python scripts/kaggle_infer.py run --url URL1 --url URL2      several episodes in one GPU session
  python scripts/kaggle_infer.py run --transcript-file FILE --title T
  python scripts/kaggle_infer.py run --audio-file episode.mp3       uploaded as the private pod2post-uploads dataset

Each run pays Kaggle's start-up once (installs + model load, ~10 min), so batch episodes together.
Results: runs/<date>/07_infer-v<N>/jobs/<id>/{result.json, article.md}; GPU time goes into
runs/quota_log.csv like every other kernel. The website's "Kaggle GPU" models call run_jobs().
Kaggle credentials are read only by the kaggle CLI (see scripts/kaggle_run.py).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_notebook as nb  # noqa: E402
import kaggle_run as kr  # noqa: E402

KERNEL = "07_infer"
DEPS = "07_deps"  # vLLM pre-installed as a tar that 07_infer attaches; saves ~4 min of every run
ADAPTERS = {"sft-dpo": "runs/2026-10-23/04_dpo-v4/adapter",  # the adapters the final eval scored
            "sft": "runs/2026-10-22/03_sft-v2/adapter"}
ADAPTER_FILES = ["adapter_config.json", "adapter_model.safetensors"]
STAGE = kr.REPO / kr.DATASETS["adapters"][1]
UPLOADS = "pod2post-uploads"  # audio files for one run; each run replaces the previous version
POLL_S = 30


def stage_adapters() -> None:
    """Flat copies '<adapter>__<file>' (Kaggle datasets skip sub-folders); 07_infer rebuilds the folders."""
    STAGE.mkdir(parents=True, exist_ok=True)
    for name, folder in ADAPTERS.items():
        for f in ADAPTER_FILES:
            src, dst = kr.REPO / folder / f, STAGE / f"{name}__{f}"
            if not dst.is_file() or kr.sha256_file(dst) != kr.sha256_file(src):
                kr.copy_verified(src, dst)


def upload_adapters() -> int:
    stage_adapters()
    code = kr.cmd_dataset(argparse.Namespace(name="adapters", message=None, force=False))
    return code if code else (0 if kr.wait_dataset_ready("adapters") else 1)


def upload_audio(jobs: list[dict]) -> None:
    """Jobs with a local "audio_path" -> one private dataset version holding those files (older versions
    deleted, so listeners' audio does not pile up); each job gets "audio", its file name in the dataset."""
    stage = kr.REPO / "build" / "dataset" / UPLOADS
    shutil.rmtree(stage, ignore_errors=True)
    stage.mkdir(parents=True)
    for job in jobs:
        src = Path(job.pop("audio_path"))
        job["audio"] = f"{uuid.uuid4().hex[:12]}{src.suffix.lower() or '.mp3'}"  # unique: proves the version
        try:  # a hard link: on Cloud Run the disk is memory, and uploads can be hundreds of MB
            os.link(src, stage / job["audio"])
        except OSError:
            shutil.copyfile(src, stage / job["audio"])
    ref = f"{kr.username()}/{UPLOADS}"
    (stage / "dataset-metadata.json").write_text(json.dumps(
        {"title": UPLOADS, "id": ref, "licenses": [{"name": "other"}]}), encoding="utf-8")
    new = kr.dataset_status(ref) == "absent"
    cmd = (["datasets", "create", "-p", str(stage), "-q"] if new
           else ["datasets", "version", "-p", str(stage), "-q", "-d", "-m", "audio for one 07_infer run"])
    out = kr.run_cli(cmd, timeout=1800)
    shutil.rmtree(stage, ignore_errors=True)
    # The CLI's exit code is not reliable here (it has failed on uploads that went through), so success
    # means: the dataset is ready and lists this run's files.
    names = {j["audio"] for j in jobs}
    deadline = time.time() + (900 if out.returncode == 0 else 180)
    while True:
        listing = kr.run_cli_read(["datasets", "files", ref]).stdout
        if kr.dataset_status(ref) == "ready" and all(n in listing for n in names):
            return
        if time.time() > deadline:
            raise RuntimeError(f"audio upload to Kaggle failed: {(out.stdout + out.stderr).strip()[-300:]}")
        time.sleep(15)


def build_with_jobs(jobs: list[dict], username: str, out_root: Path = nb.BUILD_DIR) -> tuple[Path, str]:
    """07_infer's notebook with a JOBS cell after the config cell; returns (folder, code hash).
    Jobs with uploaded audio also attach the pod2post-uploads dataset."""
    out_dir, meta, digest = nb.build(KERNEL, username, smoke=False, out_root=out_root)
    if any(j.get("audio") for j in jobs):
        meta["dataset_sources"].append(f"{username}/{UPLOADS}")
        (out_dir / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    book_path = out_dir / "run.ipynb"
    book = json.loads(book_path.read_text(encoding="utf-8"))
    book["cells"].insert(2, {"cell_type": "code", "id": "cell-jobs", "metadata": {}, "execution_count": None,
                             "outputs": [], "source": ["# Jobs (generated by scripts/kaggle_infer.py)\n",
                                                       f"JOBS = {jobs!r}\n"]})
    book_path.write_text(json.dumps(book, indent=1), encoding="utf-8")
    return out_dir, digest


def push_built(kernel: str, out_dir: Path, digest: str, timeout_s: int, **record) -> str:
    """Push a built kernel folder and record the run like kaggle_run.py does; returns the kernel ref."""
    ref = kr.kernel_ref(kernel, full=True)
    status, _ = kr.kernel_status(ref)
    if status in kr.ACTIVE:
        raise RuntimeError(f"{ref} is already {status}; wait for it to finish first.")
    out = kr.run_cli(["kernels", "push", "-p", str(out_dir), "-t", str(timeout_s)], timeout=600)
    text = (out.stdout + out.stderr).strip()
    m = kr.PUSH_OK_RE.search(text)
    if out.returncode != 0 or not m:
        raise RuntimeError(f"Push of {ref} failed: {text[-500:]}")
    state = kr.load_state()
    state["runs"][ref.split("/", 1)[1]] = {
        "kernel": kernel, "mode": "full", "version": int(m[1]), "code_hash": digest,
        "pushed_at": kr.utcnow().isoformat(), "status": "queued", "logged": False, "resume": False, **record,
    }
    kr.save_state(state)
    if kr.INVALID_SOURCES_RE.search(text):
        raise RuntimeError(f"Kaggle rejected some inputs of {ref}: {text[-300:]}")
    return ref


def push(jobs: list[dict], timeout_s: int) -> str:
    """Build 07_infer with a JOBS cell and push it; returns the kernel ref."""
    dataset = f"{kr.username()}/{kr.DATASETS['adapters'][0]}"
    if kr.dataset_status(dataset) != "ready":
        raise RuntimeError(f"{dataset} is not ready; run `python scripts/kaggle_infer.py adapters` first.")
    deps, _ = kr.kernel_status(kr.kernel_ref(DEPS, full=True))
    if deps != "complete":
        raise RuntimeError(f"pod2post-07-deps is {deps}; run `python scripts/kaggle_infer.py deps` first.")
    out_dir, digest = build_with_jobs(jobs, kr.username())
    return push_built(KERNEL, out_dir, digest, timeout_s, jobs=[j["id"] for j in jobs])


def build_deps() -> int:
    """Run 07_deps (CPU, no GPU quota) and wait for it."""
    out_dir, _, digest = nb.build(DEPS, kr.username(), smoke=False)
    ref = push_built(DEPS, out_dir, digest, 3600)
    status = wait(ref, 5400)
    if status != "complete":
        kr.print_error_log(ref)
        return 1
    manifest = kr.read_manifest(ref, kr.REPO / "build" / "deps-manifest")
    print(f"{ref} complete: {json.dumps({k: manifest.get(k) for k in ('python', 'tar_gb')}) if manifest else ''}")
    return 0


def wait(ref: str, timeout_s: int, on_status=None) -> str:
    pushed, start, seen_active, last = time.time(), time.time(), False, None
    while True:
        status, failure = kr.kernel_status(ref)
        seen_active |= status in kr.ACTIVE
        if status != last:
            print(f"[{kr.utcnow():%H:%M:%S}Z] {ref}: {status}", flush=True)
            last = status
        if on_status:
            on_status(status, time.time() - start)
        stale = status in kr.TERMINAL and not seen_active and time.time() - pushed < kr.PUSH_GRACE_S
        if (status in kr.TERMINAL or status == "absent") and not stale:
            kr.record_status(kr.load_state(), ref, status, failure)
            return status
        if time.time() - start > timeout_s:
            return status
        time.sleep(15 if stale else POLL_S)


def pull(ref: str) -> Path:
    if kr.cmd_pull(argparse.Namespace(kernel=KERNEL, full=True)) != 0:
        raise RuntimeError(f"could not download the output of {ref}")
    return kr.REPO / kr.load_state()["runs"][ref.split("/", 1)[1]]["pulled_to"]


def run_jobs(jobs: list[dict], on_status=None) -> dict[str, dict]:
    """Push, wait, pull; returns {job id: result dict} (the website's generate() keys, or {"error"}).
    The kernel is allowed 30 min plus 15 min per episode."""
    timeout_s = min(3 * 3600, 1800 + 900 * len(jobs))
    if any(j.get("audio_path") for j in jobs):
        if on_status:
            on_status("uploading the audio to Kaggle", 0)
        upload_audio([j for j in jobs if j.get("audio_path")])
    ref = push(jobs, timeout_s)
    status = wait(ref, timeout_s + 1800, on_status)
    if status != "complete":
        kr.print_error_log(ref)
        raise RuntimeError(f"Kaggle run {ref} ended as {status!r}; see `python scripts/kaggle_run.py logs {KERNEL} --full`.")
    dest = pull(ref)
    results = {}
    for job in jobs:
        path = dest / "jobs" / job["id"] / "result.json"
        res = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {"error": "no result file"}
        if res.get("file"):
            res["file"] = str(path.parent / res["file"])
        results[job["id"]] = res
    return results


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("adapters", help="upload the trained adapters as a private Kaggle dataset")
    sub.add_parser("deps", help="pre-install vLLM for 07_infer (CPU kernel, no GPU quota)")
    r = sub.add_parser("run", help="write articles on Kaggle")
    r.add_argument("--url", action="append", default=[], help="podcast link (mp3, RSS feed or episode page); repeatable")
    r.add_argument("--transcript-file", action="append", default=[], help="transcript text file; repeatable")
    r.add_argument("--audio-file", action="append", default=[], help="local audio file; repeatable")
    r.add_argument("--episode", default="", help="feed links: part of the episode title (default: newest)")
    r.add_argument("--title", default="")
    r.add_argument("--keyword", default="")
    r.add_argument("--model", choices=["sft_dpo", "sft", "base"], default="sft_dpo")  # kaggle_jobs.MODEL_DIRS
    args = ap.parse_args(argv)
    if args.cmd == "adapters":
        return upload_adapters()
    if args.cmd == "deps":
        return build_deps()
    common = {"episode": args.episode, "title": args.title, "keyword": args.keyword, "model": args.model}
    jobs = [{"id": f"j{i}", "url": u, "transcript": "", **common} for i, u in enumerate(args.url, 1)]
    jobs += [{"id": f"j{len(jobs) + i}", "url": "", "transcript": Path(f).read_text(encoding="utf-8"), **common}
             for i, f in enumerate(args.transcript_file, 1)]
    jobs += [{"id": f"j{len(jobs) + i}", "url": "", "transcript": "", "audio_path": f, **common}
             for i, f in enumerate(args.audio_file, 1)]
    if not jobs:
        ap.error("give at least one --url or --transcript-file")
    results = run_jobs(jobs)
    for job in jobs:
        res = results[job["id"]]
        print(f"\n=== {job['id']} {job['url'] or job.get('audio') or '(transcript)'}")
        print(res.get("error") or f"{res['title']}\n{res['file']}\n{res['scores']}")
    return 0 if all("error" not in r for r in results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
