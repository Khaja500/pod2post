# %% [markdown]
# # pod2post · 02_teacher_gen — Qwen3-14B-AWQ on vLLM across both T4s
# Distillation data: Stage A notes, Stage B articles + social packs, generic rewrites for DPO.
# SMOKE: 3 episodes. Full: ~520 episodes; stops starting new batches at 2h40m, and a rerun
# with `kaggle_run.py push 02_teacher_gen --full --resume` continues from the saved records.

# %%
import json
import os

os.environ.setdefault("HF_HOME", "/tmp/hf")  # model weights stay out of /kaggle/working

import yaml

from src import kaggle_env as K

run = K.start_run(KERNEL, SMOKE, CODE_HASH)
K.install_requirements(BUNDLE_DIR / "requirements-kaggle.txt", only=["vllm"])
# vllm brings torch 2.13 (CUDA 13.0) but keeps Kaggle's torchaudio 2.11+cu128, which then refuses to
# import, and transformers imports torchaudio whenever it is installed. Text-only kernels don't need it.
K.pip_uninstall(["torchaudio"])

# %%
from src.data import teacher

cfg = yaml.safe_load((BUNDLE_DIR / "configs" / "teacher.yaml").read_text())
resume_dir = K.find_upstream(SLUG) / "teacher" if RESUME else None
stats = teacher.run(K.find_upstream(DATA), K.WORKING / "teacher", cfg, smoke=bool(SMOKE),
                    resume_dir=resume_dir)
print(json.dumps(stats, indent=2))

# %%
K.finish_run(run, teacher=stats, gpu=K.gpu_info())
