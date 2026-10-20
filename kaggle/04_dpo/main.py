# %% [markdown]
# # pod2post · 04_dpo — Phase 0 placeholder
# Real job: DPO from the SFT adapter (Phase 6). Until then this stub proves kernel chaining:
# it reads each upstream kernel's run manifest, writes a checkpoint for the resume test,
# and writes its own run manifest.

# %%
from src import kaggle_env as K

K.placeholder_stage(CFG, write_checkpoint=True)
