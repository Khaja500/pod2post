# %% [markdown]
# # pod2post · 03_sft — Phase 0 placeholder
# Real job: QLoRA SFT (Phase 5). Until then this stub proves kernel chaining:
# it reads each upstream kernel's run manifest, writes a checkpoint for the resume test,
# and writes its own run manifest.

# %%
from src import kaggle_env as K

K.placeholder_stage(CFG, write_checkpoint=True)
