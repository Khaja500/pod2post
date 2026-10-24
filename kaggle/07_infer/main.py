# %% [markdown]
# # pod2post · 07_infer — podcast links / transcripts -> articles on Kaggle's GPUs
# Inference only, sent by `scripts/kaggle_infer.py`: the trained adapters come from the private
# pod2post-adapters dataset, JOBS (a generated cell) lists the episodes, and listeners' uploads come in the
# pod2post-uploads dataset. Links are downloaded and
# transcribed first, in a subprocess on Kaggle's own torch (faster-whisper needs CUDA 12); then vLLM, which
# installs its own torch (pre-installed by 07_deps), writes every article on both T4s with the website's code
# (src/infer/article.py).

# %%
import json
import os
import subprocess
import sys

os.environ.setdefault("HF_HOME", "/tmp/hf")

from src import kaggle_env as K

run = K.start_run(KERNEL, SMOKE, CODE_HASH)
jobs_file = K.write_json("jobs.json", JOBS)
K.install_requirements(BUNDLE_DIR / "requirements-kaggle.txt", only=["faster-whisper", "av"])
if K.ON_KAGGLE:  # episode pages (YouTube, SoundCloud, ...); the site pins the same version
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "yt-dlp==2026.10.21"], check=True)

# %%
uploads = [str(K.find_upstream("pod2post-uploads"))] if any(j.get("audio") for j in JOBS) else []
if any((j.get("url") or "").strip() or j.get("audio") for j in JOBS):
    subprocess.run([sys.executable, "-m", "src.infer.kaggle_jobs", "transcribe", str(jobs_file),
                    str(K.WORKING / "jobs"), *uploads], check=True, cwd=BUNDLE_DIR)

# %%
from pathlib import Path

from src.infer import kaggle_jobs

# Kaggle's torchaudio is built for its own torch's CUDA and breaks vllm's (see 02_teacher_gen); transformers
# imports it whenever it is installed. Removed before 07_deps' packages go on the path: with them on
# PYTHONPATH the uninstall did not take effect (07_infer-v3).
K.pip_uninstall(["torchaudio"])
if K.ON_KAGGLE:
    import importlib.util

    assert importlib.util.find_spec("torchaudio") is None, "torchaudio is still installed"
# vLLM from 07_deps' tar (~2 min faster than pip); pip if it is not attached or does not import.
deps = next(Path(K.INPUT).rglob(kaggle_jobs.ENV_TAR), None) if K.ON_KAGGLE else None
if not kaggle_jobs.use_prebuilt_env(deps):
    K.install_requirements(BUNDLE_DIR / "requirements-kaggle.txt", only=kaggle_jobs.ENV_PACKAGES)

# %%
import yaml

cfg = yaml.safe_load((BUNDLE_DIR / "configs" / "eval.yaml").read_text())
adapters = kaggle_jobs.unpack_adapters(K.find_upstream("pod2post-adapters"), Path("/tmp/adapters"))
summary = kaggle_jobs.write_all(JOBS, K.WORKING / "jobs", adapters, cfg)
print(json.dumps(summary, indent=2))

# %%
K.finish_run(run, jobs=summary, gpu=K.gpu_info())
