# %% [markdown]
# # pod2post · 01_transcribe — faster-whisper on one T4
# SMOKE: first 2 minutes of one episode. Full: 5 Creative-Commons episodes plus the
# Changelog WER reference. Transcripts land in transcripts/<id>.json (episode schema).

# %%
import json

import yaml

from src import kaggle_env as K

K.pin_single_gpu()
run = K.start_run(KERNEL, SMOKE, CODE_HASH)
K.install_requirements(BUNDLE_DIR / "requirements-kaggle.txt", only=["faster-whisper", "av"])

# %%
from src.asr.transcribe import transcribe_all

cfg = yaml.safe_load((BUNDLE_DIR / "configs" / "asr.yaml").read_text())
report = transcribe_all(K.find_upstream("pod2post-audio"), K.WORKING / "transcripts", cfg, smoke=bool(SMOKE))
print(json.dumps(report, indent=2))

# %%
K.finish_run(run, asr=report, gpu=K.gpu_info())
