# pod2post

Podcast episode → SEO article + social pack, using a QLoRA-post-trained (SFT + DPO) ~3B open model trained on Kaggle's free T4 GPUs. `spec.md` defines what gets built and `plan.md` sets the order of work. Non-obvious choices are recorded in `docs/DECISIONS.md`.

## Setup (once)

1. Verify your phone number on Kaggle. GPU and internet access in notebooks require it.
2. Give the Kaggle CLI credentials using **one** of these options:
   - `kaggle auth login` (OAuth in the browser; nothing to copy), or
   - kaggle.com → Settings → API → **Generate New Token**, then save the token to `~/.kaggle/access_token` and run `chmod 600 ~/.kaggle/access_token`, or
   - the legacy `kaggle.json` at `~/.kaggle/kaggle.json` (`chmod 600`).
3. `python3 -m venv .venv && .venv/bin/pip install -r requirements.txt`
4. `python scripts/kaggle_run.py check`

Never paste a key into chat or commit one. The secret file names are git-ignored and `check` flags any that get tracked.

## Run the whole project

```
python3 scripts/kaggle_run.py pipeline          # every stage in SMOKE mode (tiny data, minutes per kernel)
python3 scripts/kaggle_run.py pipeline --full   # the real thing: ~10-15 GPU-hours on Kaggle T4s
python3 scripts/kaggle_run.py pipeline --full --from 03_sft   # pick up from a stage
```

`pipeline` runs 01 transcribe → 02 teacher generation → 03 SFT → 04 DPO → 05 evaluation → 06 LLM judge on Kaggle. It also does the local steps in between:
- **After 02:** filter → build SFT/DPO data → upload the dataset. In full mode it also checks the Phase 3 acceptance rule.
- **After 01:** WER is computed (full mode).
- **After 05 and 06:** `reports/results.md` is written.

If a long run stops at its time budget, the pipeline resumes it automatically. In full mode, a stage whose current code has never passed a smoke run gets one first. Network blips are retried rather than ending the run.

## Running kernels one at a time

```
python scripts/kaggle_run.py push 00_smoke    # GPU / install / environment check (~10 min)
python scripts/kaggle_run.py wait 00_smoke
python scripts/kaggle_run.py pull 00_smoke    # -> runs/<date>/00_smoke-v<N>/
python scripts/kaggle_run.py pipeline         # 01 -> 05 in SMOKE mode, stops on first error
python scripts/kaggle_run.py push 03_sft --full   # only after a passing smoke run of the same code
python scripts/kaggle_run.py quota
```

All commands target the **smoke** kernel (`<slug>-smoke`, `SMOKE=1`) unless `--full` is given. `push` refuses to start in any of these cases unless `--force` is passed:
- the kernel is already queued or running
- an upstream kernel hasn't completed
- a full run is requested before a smoke run of the same code
- a full run is requested while another run is active

Every finished run is appended to `runs/quota_log.csv`. Run state lives in `runs/state.json`. Both files are committed so a fresh session can pick up where the last one stopped.

## How the Kaggle notebooks are made

Kernel logic lives in `kaggle/<kernel>/main.py` (percent-format cells) and `src/`. `scripts/build_notebook.py` renders `build/kaggle/<slug>/run.ipynb` and `kernel-metadata.json`, bundling `src/`, `configs/` and `requirements-kaggle.txt` into the notebook. Don't edit notebooks on kaggle.com; edit the repo and push again.

`pytest` builds every kernel and runs the notebooks in pipeline order against a simulated `/kaggle` tree, which catches broken kernels before they spend GPU quota.

## Claude Code on the web

In the cloud environment:
- Network: set access to custom and allow `api.kaggle.com`, `www.kaggle.com`, `kaggle.com` and `storage.googleapis.com`, plus the default package registries.
- Env vars: `KAGGLE_API_TOKEN`, or `KAGGLE_USERNAME` + `KAGGLE_KEY`. Setting `KAGGLE_USERNAME` is harmless and lets the scripts skip a lookup.

Then run `python scripts/kaggle_run.py check`.

## Layout

```
configs/   sft.yaml dpo.yaml teacher.yaml eval.yaml
data/      SOURCES.md (raw/ and processed/ are git-ignored)
src/       kaggle_env.py (kernel runtime helpers) + asr/ data/ train/ infer/ eval/
scripts/   build_notebook.py kaggle_run.py
kaggle/    00_smoke 01_transcribe 02_teacher_gen 03_sft 04_dpo 05_eval
runs/      state.json quota_log.csv (+ pulled outputs, git-ignored)
docs/      DECISIONS.md
tests/
```

## Data (Phase 1)

```
.venv/bin/python -m src.data.download    # sources -> data/processed/episodes.jsonl, freezes the 50-episode test set
.venv/bin/python -m src.data.chunk       # -> data/processed/chunks.jsonl (<=1,500 tokens, 150 overlap)
python3 scripts/kaggle_run.py dataset    # upload as the private Kaggle dataset pod2post-data
```

Sources and licenses are listed in `data/SOURCES.md`. The test-set IDs are stored as hashes in `data/processed/test_ids.sha256`, and training code refuses to run on them.

## Demo (Phase 8)

```
.venv/bin/pip install gradio
python3 app/gradio_app.py --adapter runs/<date>/04_dpo-v<N>/adapter --backend hf   # local: CPU / Apple GPU (slow)
```

In a Kaggle GPU notebook with the `pod2post-04-dpo` output attached, use the fast vLLM backend and a public link:

```
!git clone <this repo> pod2post && cd pod2post && pip install -q vllm==0.32.0 gradio && pip uninstall -y -q torchaudio
!python app/gradio_app.py --adapter /kaggle/input/pod2post-04-dpo/adapter --share
```

Paste a transcript (`Speaker: text` lines or plain paragraphs) or upload a short audio clip, which faster-whisper transcribes. The demo returns the article, the social pack, the SEO checks, and any names or numbers the article uses that are not in the transcript. It runs the same pipeline code that the evaluation scored.
