# %% [markdown]
# # pod2post · 04_dpo — DPO from the SFT adapter
# 1. Sample the SFT model on the DPO prompts (vLLM on both T4s, in its own venv so its torch
#    never meets the training stack), then score every sample and the teacher's answer with the
#    eval's faithfulness scorer (D39).
# 2. Pair the least faithful sample with the most faithful clean answer, add synthetic pairs
#    as a third of the total, then DPO on one T4 with the SFT adapter as reference.
# A --resume run reuses the previous run's scored samples. It continues from the previous
# checkpoint only if that run was cut off (no adapter saved); after a finished run it trains afresh.

# %%
import json
import os
import shutil
import subprocess
import sys
import time

os.environ.setdefault("HF_HOME", "/tmp/hf")

import yaml

from src import kaggle_env as K

run = K.start_run(KERNEL, SMOKE, CODE_HASH)
K.install_requirements(BUNDLE_DIR / "requirements-kaggle.txt", only=K.TRAIN_STACK)

# %%
data = K.find_upstream(DATA)
sft_adapter = K.find_upstream(next(s for s in UPSTREAM if "-03-sft" in s)) / "adapter"
cfg = yaml.safe_load((BUNDLE_DIR / "configs" / "dpo.yaml").read_text())
eval_cfg = yaml.safe_load((BUNDLE_DIR / "configs" / "eval.yaml").read_text())
previous = K.find_upstream(SLUG) if RESUME else None
scored_path = K.WORKING / "sft_scored.jsonl"

from src.data.build_dpo import assemble, sampling_order
from src.data.chunk import read_jsonl, write_jsonl

prompts = list(read_jsonl(data / "dpo_synthetic.jsonl"))
if previous is not None and (previous / scored_path.name).is_file():
    shutil.copy(previous / scored_path.name, scored_path)
else:
    VENV = "/tmp/vllm-venv"
    if not os.path.exists(f"{VENV}/bin/python"):  # Kaggle's Debian python has no ensurepip, so no `-m venv`
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "virtualenv"], check=True)
        subprocess.run([sys.executable, "-m", "virtualenv", "-q", VENV], check=True)
        subprocess.run([f"{VENV}/bin/pip", "install", "-q", "pyyaml", "-r", str(K.filtered_requirements(
            BUNDLE_DIR / "requirements-kaggle.txt", ["vllm", "textstat", "spacy", "en_core_web_sm"]))], check=True)
        subprocess.run([f"{VENV}/bin/pip", "uninstall", "-y", "-q", "torchaudio"], check=True)  # see 02_teacher_gen
    venv_env = {**os.environ, "PYTHONPATH": str(BUNDLE_DIR)}
    order_path, samples_path = K.WORKING / "sample_prompts.jsonl", K.WORKING / "sft_samples.jsonl"
    write_jsonl(order_path, sampling_order(prompts, cfg["seed"]))
    subprocess.run([f"{VENV}/bin/python", "-m", "src.train.sample_sft", str(order_path), str(sft_adapter),
                    str(samples_path), eval_cfg["base_model"], str(cfg["sample_time_budget_s"]),
                    *([str(cfg["smoke"]["num_sampled"])] if SMOKE else [])],
                   check=True, cwd=BUNDLE_DIR, env=venv_env)
    subprocess.run([f"{VENV}/bin/python", "-m", "src.train.score_samples", str(samples_path), str(order_path),
                    str(data / "chunks.jsonl"), str(scored_path), eval_cfg["faithfulness"]["nli_model"]],
                   check=True, cwd=BUNDLE_DIR, env=venv_env)

# %%
K.pin_single_gpu()
from src.train import dpo

scored = {r["episode_id"]: r for r in read_jsonl(scored_path)}
pairs = assemble(prompts, scored, seed=cfg["seed"], margin=cfg["faith_margin"], synthetic_share=cfg["synthetic_share"],
                 min_pairs=cfg["smoke"]["num_examples"] + 2 if SMOKE else 0)  # smoke data has only 3 sampled prompts
write_jsonl(K.WORKING / "dpo_pairs.jsonl", pairs)
left = cfg["kernel_budget_s"] - (time.time() - run["started_epoch"])  # sampling time varies; training gets the rest
cfg["time_budget_s"] = min(cfg["time_budget_s"], left)
finished = previous is not None and (previous / "adapter").is_dir()  # its checkpoint ran a different schedule
resume = K.latest_checkpoint(previous) if previous is not None and not finished else None
metrics = dpo.train(pairs, data, sft_adapter, K.WORKING, cfg, smoke=bool(SMOKE), resume_from=resume)
metrics["sampled_prompts"] = len(scored)
metrics["training_budget_s"] = round(cfg["time_budget_s"])
print(json.dumps(metrics, indent=2))

# %%
K.finish_run(run, dpo=metrics, gpu=K.gpu_info())
