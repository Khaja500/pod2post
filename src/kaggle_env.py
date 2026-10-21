"""Runtime helpers shared by every Kaggle kernel.

On Kaggle the paths are /kaggle/input and /kaggle/working. Locally, set
POD2POST_KAGGLE_ROOT to a fake root to run a built notebook end-to-end
(tests/test_build_notebook.py does this). Kernels never need secrets.
"""
from __future__ import annotations

import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ON_KAGGLE = "POD2POST_KAGGLE_ROOT" not in os.environ and Path("/kaggle/working").is_dir()
ROOT = Path(os.environ.get("POD2POST_KAGGLE_ROOT", "/kaggle"))
INPUT = ROOT / "input"
WORKING = ROOT / "working"
MANIFEST = "run_manifest.json"
TRAIN_STACK = ["unsloth", "unsloth_zoo", "transformers", "trl", "peft", "accelerate", "bitsandbytes", "datasets"]
CHECKPOINT_RE = re.compile(r"checkpoint-(\d+)")

# Unsloth writes its compiled trainers to ./unsloth_compiled_cache (= /kaggle/working),
# which would ship with every kernel output. Kernels import this module before unsloth.
os.environ.setdefault("UNSLOTH_COMPILE_LOCATION", "/tmp/unsloth_compiled_cache")


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def pin_single_gpu(index: int = 0) -> None:
    """Expose one T4 of the 2xT4 machine (spec §2). Must run before torch is imported."""
    if "torch" in sys.modules:
        raise RuntimeError("pin_single_gpu() must be called before torch is imported")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(index)


def write_json(rel: str, obj) -> Path:
    path = WORKING / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")
    return path


def write_text(rel: str, text: str) -> Path:
    path = WORKING / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


# --- run manifest (read back by scripts/kaggle_run.py for the quota log) ---

def start_run(kernel: str, smoke: int, code_hash: str) -> dict:
    if ON_KAGGLE:  # libraries that write caches to the cwd (Unsloth's tokenizer cache) stay out of outputs
        scratch = Path("/tmp/pod2post-cwd")
        scratch.mkdir(exist_ok=True)
        os.chdir(scratch)
    run = {
        "kernel": kernel,
        "smoke": bool(smoke),
        "code_hash": code_hash,
        "status": "running",
        "started_at": utcnow(),
        "started_epoch": time.time(),
        "finished_at": None,
        "runtime_s": None,
    }
    write_json(MANIFEST, run)
    return run


def finish_run(run: dict, **extra) -> dict:
    run.update(extra)
    run.update(
        status="complete",
        finished_at=utcnow(),
        runtime_s=round(time.time() - run["started_epoch"], 1),
    )
    write_json(MANIFEST, run)
    print(f"[kaggle_env] {run['kernel']} complete in {run['runtime_s']}s")
    return run


# --- inputs attached via kernel_sources / dataset_sources ---

def list_inputs(max_depth: int = 3) -> list[str]:
    """Directory tree under /kaggle/input, to confirm how Kaggle mounts sources."""
    if not INPUT.is_dir():
        return []
    out = []
    for dirpath, dirnames, filenames in os.walk(INPUT):
        depth = len(Path(dirpath).relative_to(INPUT).parts)
        if depth >= max_depth:
            dirnames[:] = []
        out.append(f"{dirpath}/ ({len(filenames)} files)")
    return out


def find_upstream(slug: str) -> Path:
    """Locate an attached kernel output by slug.

    Searched a few levels deep because the mount layout under /kaggle/input
    is not guaranteed to be /kaggle/input/<slug>.
    """
    for dirpath, dirnames, _ in os.walk(INPUT):
        if Path(dirpath).name == slug:
            return Path(dirpath)
        if len(Path(dirpath).relative_to(INPUT).parts) >= 3:
            dirnames[:] = []
    raise FileNotFoundError(f"upstream output '{slug}' not attached; inputs: {list_inputs()}")


def read_upstream_manifest(slug: str) -> dict:
    manifest = json.loads((find_upstream(slug) / MANIFEST).read_text(encoding="utf-8"))
    if manifest.get("status") != "complete":
        raise RuntimeError(f"upstream {slug} did not complete: {manifest.get('status')}")
    return manifest


def latest_checkpoint(*roots: Path) -> Path | None:
    """Highest-step checkpoints/checkpoint-N under any root, for resuming."""
    best = None
    for root in roots:
        for ck in Path(root).glob("**/checkpoints/checkpoint-*"):
            m = CHECKPOINT_RE.fullmatch(ck.name)
            if m and ck.is_dir() and (best is None or int(m[1]) > best[0]):
                best = (int(m[1]), ck)
    return best[1] if best else None


# --- environment ---

def gpu_info() -> dict:
    info = {"nvidia_smi": None, "torch": None}
    if shutil.which("nvidia-smi"):
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
            capture_output=True, text=True,
        )
        info["nvidia_smi"] = out.stdout.strip().splitlines()
    try:
        import torch
    except ImportError:
        return info
    cuda = torch.cuda.is_available()
    info["torch"] = {
        "version": torch.__version__,
        "cuda": torch.version.cuda,
        "available": cuda,
        "visible_devices": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        # T4 is sm_75: no native bf16, so training must use fp16 (spec §2).
        "capability": list(torch.cuda.get_device_capability(0)) if cuda else None,
        "native_bf16": cuda and torch.cuda.get_device_capability(0) >= (8, 0),
    }
    return info


def env_report() -> dict:
    disk = shutil.disk_usage(WORKING if WORKING.exists() else "/")
    return {
        "time": utcnow(),
        "on_kaggle": ON_KAGGLE,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "gpu": gpu_info(),
        "working_free_gb": round(disk.free / 1e9, 1),
        "inputs": list_inputs(),
    }


def fp16_matmul_tflops(n: int = 4096, iters: int = 20) -> float | None:
    """Quick fp16 throughput check; None when no CUDA device."""
    try:
        import torch
    except ImportError:
        return None
    if not torch.cuda.is_available():
        return None
    a = torch.randn(n, n, device="cuda", dtype=torch.float16)
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(iters):
        a @ a
    torch.cuda.synchronize()
    return round(2 * n**3 * iters / (time.time() - t0) / 1e12, 2)


def filtered_requirements(req: Path, only: list[str]) -> Path:
    """Write the pinned lines for just the `only` packages to a temp requirements file."""
    wanted = {re.sub(r"[-_.]+", "-", name.lower()) for name in only}
    lines = [line for line in Path(req).read_text().splitlines()
             if re.sub(r"[-_.]+", "-", re.split(r"[=<>@;\s]", line.strip(), 1)[0].lower()) in wanted]
    if len(lines) != len(wanted):
        raise ValueError(f"{sorted(wanted)} not all pinned in {req}")
    out = Path(f"/tmp/pod2post-requirements-{'-'.join(sorted(wanted))[:60]}.txt")
    out.write_text("\n".join(lines) + "\n")
    return out


def install_requirements(req: Path, only: list[str] | None = None) -> None:
    """pip install the pinned Kaggle requirements (or just the `only` packages). Skipped off Kaggle."""
    if only:
        req = filtered_requirements(req, only)
    if not ON_KAGGLE:
        print(f"[kaggle_env] not on Kaggle; skipping pip install -r {req}")
        return
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", str(req)], check=True)


def pip_uninstall(packages: list[str]) -> None:
    """Remove preinstalled packages that clash with what a kernel installed. Skipped off Kaggle."""
    if ON_KAGGLE:
        subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "-q", *packages], check=True)


def import_versions(modules: list[str]) -> dict:
    """Import each module (in order) and return its version. Skipped off Kaggle."""
    if not ON_KAGGLE:
        return {}
    versions = {}
    for name in modules:
        mod = __import__(name)
        versions[name] = getattr(mod, "__version__", "unknown")
    return versions


def pip_freeze() -> str:
    return subprocess.run(
        [sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True, check=True
    ).stdout


# --- Phase 0 placeholder stage (replaced by real code in later phases) ---

def placeholder_stage(cfg: dict, write_checkpoint: bool = False) -> dict:
    """Prove chaining: read every upstream manifest, find checkpoints, write our own outputs."""
    pin_single_gpu()
    run = start_run(cfg["KERNEL"], cfg["SMOKE"], cfg["CODE_HASH"])
    upstream = {slug: read_upstream_manifest(slug)["code_hash"] for slug in cfg["UPSTREAM"]}
    resume_from = latest_checkpoint(find_upstream(cfg["SLUG"])) if cfg["RESUME"] else None
    print(f"[kaggle_env] upstream: {upstream}  resume_from: {resume_from}")
    if write_checkpoint:
        step = int(CHECKPOINT_RE.fullmatch(resume_from.name)[1]) + 5 if resume_from else 5
        write_text(f"checkpoints/checkpoint-{step}/PLACEHOLDER", "phase 0 stub checkpoint\n")
    return finish_run(
        run,
        upstream=upstream,
        resume_from=str(resume_from) if resume_from else None,
        env=env_report(),
    )
