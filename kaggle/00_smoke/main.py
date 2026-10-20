# %% [markdown]
# # pod2post · 00_smoke — environment check
# Confirms the GPU (T4, fp16, no native bf16), internet + pinned installs, and the
# /kaggle/input layout, then writes outputs that `kaggle_run.py pull` brings back.

# %%
import json
from src import kaggle_env as K

K.pin_single_gpu()
run = K.start_run(KERNEL, SMOKE, CODE_HASH)
env = K.env_report()
K.write_json("env.json", env)
print(json.dumps(env, indent=2))

# %%
tflops = K.fp16_matmul_tflops()
print(f"fp16 matmul: {tflops} TFLOPS")

# %%
# Install the pinned stack and import it (unsloth first, as it patches transformers/trl).
K.install_requirements(BUNDLE_DIR / "requirements-kaggle.txt")
versions = K.import_versions(
    ["unsloth", "transformers", "trl", "peft", "accelerate", "bitsandbytes", "datasets", "faster_whisper", "torch"]
)
print(json.dumps(versions, indent=2))
if K.ON_KAGGLE:
    K.write_text("pip_freeze.txt", K.pip_freeze())

# %%
K.write_text("hello.txt", f"pod2post smoke kernel ran at {K.utcnow()}\n")
K.finish_run(run, fp16_tflops=tflops, versions=versions)
