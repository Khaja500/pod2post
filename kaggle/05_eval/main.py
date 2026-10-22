# %% [markdown]
# # pod2post · 05_eval — every system on the 50 held-out episodes
# Base model zero-shot, plus SFT / SFT+DPO when their kernels are attached as inputs.
# Generation on one T4 (vLLM, fp16 base + LoRA), NLI faithfulness scoring on the other.
# Full: ~13 GPU-hours, so it stops starting new episode batches at 1h50m and
# `kaggle_run.py push 05_eval --full --resume` continues (the pipeline resumes automatically).

# %%
import json
import os

os.environ.setdefault("HF_HOME", "/tmp/hf")

import yaml

from src import kaggle_env as K

run = K.start_run(KERNEL, SMOKE, CODE_HASH)
K.install_requirements(BUNDLE_DIR / "requirements-kaggle.txt", only=["vllm", "textstat", "spacy", "en_core_web_sm"])
K.pip_uninstall(["torchaudio"])  # CUDA-version clash with vllm's torch; see 02_teacher_gen

# %%
from src.eval import kernel as E

cfg = yaml.safe_load((BUNDLE_DIR / "configs" / "eval.yaml").read_text())
systems = {"base": None}
for slug in UPSTREAM:
    name = "sft" if "-03-sft" in slug else "sft_dpo" if "-04-dpo" in slug else None
    if name:
        systems[name] = str(K.find_upstream(slug) / "adapter")
resume_dir = K.find_upstream(SLUG) if RESUME else None
summary = E.run(K.find_upstream(DATA), K.WORKING, cfg, systems, smoke=bool(SMOKE), resume_dir=resume_dir)
print(json.dumps(summary, indent=2))

# %%
K.finish_run(run, eval=summary["systems"], gpu=K.gpu_info())
