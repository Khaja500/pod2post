# %% [markdown]
# # pod2post · 07_deps — vLLM pre-installed for 07_infer (CPU only, no GPU quota)
# Installing vLLM took 4.1 of 07_infer's ~11 start-up minutes (07_infer-v1); unpacking this tar takes ~2. It installs the
# same pinned packages once with `pip --target` and saves them as one tar; 07_infer unpacks it instead
# (src/infer/kaggle_jobs.py use_prebuilt_env) and falls back to pip if the tar is missing or does not
# import. Rebuild it (`scripts/kaggle_infer.py deps`) when requirements-kaggle.txt changes.

# %%
import platform
import subprocess
import sys
import tarfile
from pathlib import Path

from src import kaggle_env as K
from src.infer.kaggle_jobs import ENV_PACKAGES, ENV_TAR

run = K.start_run(KERNEL, SMOKE, CODE_HASH)
req = K.filtered_requirements(BUNDLE_DIR / "requirements-kaggle.txt", ENV_PACKAGES)
target = Path("/tmp/vllm-env")
if K.ON_KAGGLE:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--target", str(target), "-r", str(req)], check=True)
    with tarfile.open(K.WORKING / ENV_TAR, "w") as tar:  # one file: outputs with many files upload slowly
        tar.add(target, arcname="vllm-env")

# %%
size = (K.WORKING / ENV_TAR).stat().st_size if (K.WORKING / ENV_TAR).is_file() else 0
K.finish_run(run, python=sys.version.split()[0], platform=platform.platform(), tar_gb=round(size / 1e9, 2),
             requirements=req.read_text().split())
