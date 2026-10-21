# %% [markdown]
# # pod2post · 06_judge — pairwise LLM judge vs the base model (spec §6)
# The teacher (Qwen3-14B-AWQ, both T4s) compares each system's article with the base model's,
# in both A/B orders, using transcript passages retrieved for the two articles as evidence.

# %%
import json
import os
from collections import defaultdict

os.environ.setdefault("HF_HOME", "/tmp/hf")

import yaml

from src import kaggle_env as K

run = K.start_run(KERNEL, SMOKE, CODE_HASH)
K.install_requirements(BUNDLE_DIR / "requirements-kaggle.txt", only=["vllm"])
K.pip_uninstall(["torchaudio"])  # CUDA-version clash with vllm's torch; see 02_teacher_gen

# %%
from src.data.chunk import read_jsonl
from src.eval import judge
from src.infer.generator import VLLMGenerator

m = yaml.safe_load((BUNDLE_DIR / "configs" / "teacher.yaml").read_text())["open_model"]
gen = VLLMGenerator(m["model"], tensor_parallel_size=m["tensor_parallel_size"], max_model_len=m["max_model_len"],
                    gpu_memory_utilization=m["gpu_memory_utilization"])
chunks = defaultdict(list)
for c in read_jsonl(K.find_upstream(DATA) / "chunks.jsonl"):
    if c["split"] == "test":
        chunks[c["episode_id"]].append(c["text"])
summary = judge.run(K.find_upstream(UPSTREAM[0]), chunks, gen, K.WORKING)
print(json.dumps(summary, indent=2))

# %%
K.finish_run(run, judge=summary, gpu=K.gpu_info())
