# %% [markdown]
# # pod2post · 04_dpo — DPO from the SFT adapter on one T4
# 1. Sample the SFT model on the DPO prompts (vLLM, in its own venv so its torch never meets
#    the training stack) and keep failing outputs = rejected type (a).
# 2. Assemble ~300 pairs mixing the four rejected types, then DPO with the SFT adapter as reference.

# %%
import json
import os
import subprocess
import sys

os.environ.setdefault("HF_HOME", "/tmp/hf")

import yaml

from src import kaggle_env as K

run = K.start_run(KERNEL, SMOKE, CODE_HASH)
K.install_requirements(BUNDLE_DIR / "requirements-kaggle.txt", only=K.TRAIN_STACK)
VENV = "/tmp/vllm-venv"
if not os.path.exists(f"{VENV}/bin/python"):
    subprocess.run([sys.executable, "-m", "venv", VENV], check=True)
    subprocess.run([f"{VENV}/bin/pip", "install", "-q", "pyyaml", "-r", str(K.filtered_requirements(
        BUNDLE_DIR / "requirements-kaggle.txt", ["vllm", "spacy", "en_core_web_sm"]))], check=True)
    subprocess.run([f"{VENV}/bin/pip", "uninstall", "-y", "-q", "torchaudio"], check=True)  # see 02_teacher_gen

# %%
data = K.find_upstream(DATA)
sft_adapter = K.find_upstream(next(s for s in UPSTREAM if "-03-sft" in s)) / "adapter"
cfg = yaml.safe_load((BUNDLE_DIR / "configs" / "dpo.yaml").read_text())
base_model = yaml.safe_load((BUNDLE_DIR / "configs" / "eval.yaml").read_text())["base_model"]
failures_path = K.WORKING / "sft_failures.jsonl"
subprocess.run([f"{VENV}/bin/python", "-m", "src.train.sample_sft", str(data / "dpo_synthetic.jsonl"),
                str(data / "chunks.jsonl"), str(sft_adapter), str(failures_path), base_model,
                *(["10"] if SMOKE else [])],
               check=True, cwd=BUNDLE_DIR, env={**os.environ, "PYTHONPATH": str(BUNDLE_DIR), "CUDA_VISIBLE_DEVICES": "0"})

# %%
from src.data.build_dpo import assemble
from src.data.chunk import read_jsonl, write_jsonl

K.pin_single_gpu()
from src.train import dpo

prompts = list(read_jsonl(data / "dpo_synthetic.jsonl"))
failures = {r["episode_id"]: r["output"] for r in read_jsonl(failures_path)}
pairs = assemble(prompts, failures, seed=cfg["seed"])
write_jsonl(K.WORKING / "dpo_pairs.jsonl", pairs)
resume = K.latest_checkpoint(K.find_upstream(SLUG)) if RESUME else None
metrics = dpo.train(pairs, data, sft_adapter, K.WORKING, cfg, smoke=bool(SMOKE), resume_from=resume)
metrics["sft_failure_prompts"] = len(failures)
print(json.dumps(metrics, indent=2))

# %%
K.finish_run(run, dpo=metrics, gpu=K.gpu_info())
