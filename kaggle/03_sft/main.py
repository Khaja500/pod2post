# %% [markdown]
# # pod2post · 03_sft — QLoRA SFT on one T4 (Unsloth + TRL, fp16)
# One LoRA adapter for both tasks (Stage A notes, Stage B article + social pack).
# SMOKE: 10 examples, 5 steps. Full: ~1,000 examples x 2 epochs; saves and stops at 2h40m,
# and `kaggle_run.py push 03_sft --full --resume` continues from the newest checkpoint.

# %%
import json

import yaml

from src import kaggle_env as K

K.pin_single_gpu()
run = K.start_run(KERNEL, SMOKE, CODE_HASH)
K.install_requirements(BUNDLE_DIR / "requirements-kaggle.txt", only=K.TRAIN_STACK)

# %%
from src.train import sft

cfg = yaml.safe_load((BUNDLE_DIR / "configs" / "sft.yaml").read_text())
resume = K.latest_checkpoint(K.find_upstream(SLUG)) if RESUME else None
metrics = sft.train(K.find_upstream(DATA), K.WORKING, cfg, smoke=bool(SMOKE), resume_from=resume)
print(json.dumps(metrics, indent=2))

# %%
K.finish_run(run, sft=metrics, gpu=K.gpu_info())
