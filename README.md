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

## Website (Phase 8)

`app/gradio_app.py` is a small website: paste a **podcast link** or upload an **audio file** (or paste a transcript), and it returns an SEO article (downloadable as `.md`), the social pack, the SEO checks, and any names or numbers the article uses that are not in the transcript. It runs the same pipeline code that the evaluation scored.

The page has an episode panel (link, audio file or transcript, with a *Try a sample transcript* button), the model picker, and the title and keyword under *Title and keyword (optional)*. Results show as score cards (SEO checks, words, keyword, X posts, unsupported facts, writing time) above tabs for the article, the social pack (readable, with the raw JSON underneath), the scores, the episode notes and the transcript used.

A podcast link can be:
- a direct audio link (`.mp3`, `.m4a`, …),
- an RSS feed: the newest episode, or the one whose title contains the text in *Episode in feed*,
- an episode page that [yt-dlp](https://github.com/yt-dlp/yt-dlp) can read (YouTube, SoundCloud, Apple Podcasts, archive.org, …). DRM-protected services such as Spotify cannot be downloaded.

Audio up to 500 MB and 3 hours (`--max-minutes`) is transcribed with faster-whisper: `large-v3-turbo` on a GPU, `--asr-model` (default `small`) on a CPU. Because the site may be shared through a public link, it only fetches http(s) URLs on public addresses and re-checks every redirect (`src/asr/url_audio.py`). Only process audio you have the right to republish.

Where the model runs:
- **Public site:** Cloud Run sends every article to the owner's Kaggle GPUs, about 12–15 minutes each ("Public site on Google Cloud Run" below).
- **Your Mac:** llama.cpp, no cloud ("Run it on a Mac"); add `--kaggle` to also offer the Kaggle GPU models.
- **A Kaggle GPU notebook**, with the `pod2post-04-dpo` output attached for the fine-tuned adapter:

```
!git clone <this repo> pod2post && cd pod2post && pip install -q vllm==0.32.0 gradio yt-dlp faster-whisper==1.2.1 av==18.1.0 && pip uninstall -y -q torchaudio
!python app/gradio_app.py --adapter /kaggle/input/pod2post-04-dpo/adapter --share
```

Open the public `gradio.live` link it prints (it lasts while the notebook runs, and the session counts against the GPU quota). Leave out `--adapter` to use the base model: in the evaluation, base was more faithful (43.5% vs 39.1%) while the fine-tuned model scored higher on SEO (96.5% vs 80.5%); see `reports/results.md`.

### Run it on a Mac (no cloud, no NVIDIA GPU)

The website runs the same model on Apple Silicon through [llama.cpp](https://github.com/ggml-org/llama.cpp): a 4-bit GGUF of the base model (2.5 GB, fits in 8 GB of RAM) with the LoRA adapter converted to GGUF and applied at run time. On an 8 GB M1 Air it reads prompts at ~100 tokens/s and writes ~13 tokens/s: a 48-minute episode (9 chunks) took 47 minutes end to end, 18 of them transcribing with Whisper `small` on the CPU and 29 writing.

```
brew install llama.cpp                       # llama-server, with Metal
.venv/bin/pip install -r requirements.txt
# 1. base model, 4-bit (Apache-2.0; pinned revision)
curl -L -o models/Qwen3-4B-Instruct-2507-Q4_K_M.gguf \
  https://huggingface.co/unsloth/Qwen3-4B-Instruct-2507-GGUF/resolve/a06e946bb6b655725eafa393f4a9745d460374c9/Qwen3-4B-Instruct-2507-Q4_K_M.gguf
# 2. the fine-tuned adapter -> GGUF, with llama.cpp's converter (needs torch + transformers;
#    use the llama.cpp source tag that matches `llama-server --version`, e.g. b11669)
git clone --depth 1 --branch b11669 https://github.com/ggml-org/llama.cpp /tmp/llama.cpp
pip install /tmp/llama.cpp/gguf-py torch transformers sentencepiece
python /tmp/llama.cpp/convert_lora_to_gguf.py --base-model-id Qwen/Qwen3-4B-Instruct-2507 --outtype f16 \
  --outfile models/pod2post-sft-dpo-lora-f16.gguf runs/<date>/04_dpo-v<N>/adapter
# 3. the website: picks llama.cpp automatically and loads every models/*-lora-*.gguf
python3 app/gradio_app.py                    # http://127.0.0.1:7860 (add --share for a public link)
```

The page then has a model picker: the fine-tuned SFT + DPO adapter or the base model. `models/` is git-ignored.

Without llama.cpp, `--backend hf` runs the 16-bit model with transformers (needs ~10 GB of free memory; slow).

### Inference on Kaggle's GPUs (no training)

`scripts/kaggle_infer.py` runs the finished models on Kaggle's 2×T4 (kernel `07_infer`): the trained adapters are stored once as the private dataset `pod2post-adapters`, and each run downloads or receives the episodes, transcribes them with Whisper `large-v3-turbo` and writes the articles with vLLM, using the website's own code (`src/infer/article.py`).

```
python scripts/kaggle_infer.py adapters                                   # once: upload the adapters
python scripts/kaggle_infer.py deps                                       # once: pre-install vLLM (CPU kernel, no GPU hours)
python scripts/kaggle_infer.py run --url <link> [--url <link2> ...]       # podcast links (mp3, RSS feed, page)
python scripts/kaggle_infer.py run --audio-file episode.mp3               # local audio (private pod2post-uploads dataset)
python3 app/gradio_app.py --kaggle                                        # the website, with "Kaggle GPU" models
```

Measured (2026-10-24, `runs/2026-10-24/07_infer-v1`): the 48-minute episode took 81 s to transcribe and 3.3 min to write (the Mac: 47 min), the 8.5-minute one 11 s and 2.4 min. A run first spends ~11 minutes starting up (installs, model load), so the two together took 17 minutes and 0.27 GPU-hours of the weekly 30. With vLLM pre-installed by `07_deps` and CUDA graphs captured only for the batch sizes the pipeline uses (D44), an uploaded 47.7-minute episode took **12.3 minutes from upload to article** through the public site (`07_infer-v4`: ~1 min upload and queue, 71 s transcription, 2 min unpacking vLLM, 2.5 min engine start, 3.2 min writing), 0.18 GPU-hours. Batch episodes into one run where possible. Each upload replaces the previous version of `pod2post-uploads`, so listeners' audio does not pile up.

### Public site on Google Cloud Run

`deploy/cloudrun/deploy.sh` builds the website image with Cloud Build and deploys it to Cloud Run with a public https link. The service runs at most one instance, scales to zero when idle (billed only while an instance is up, including ~15 idle minutes before it stops), and refuses work past `DAILY_LIMIT` articles a day.

```
PROJECT=<gcp-project> MODE=kaggle KAGGLE_USERNAME=<you> deploy/cloudrun/deploy.sh   # articles on your Kaggle GPUs
PROJECT=<gcp-project> MODE=cpu deploy/cloudrun/deploy.sh   # no GPU: the Mac set-up above (needs models/*-lora-*.gguf)
PROJECT=<gcp-project> deploy/cloudrun/deploy.sh            # one L4 GPU: vLLM with the SFT and SFT + DPO adapters
```

- **Kaggle** (`Dockerfile.kaggle`, live now): a 2-vCPU front end with no model. Every link, upload or transcript from any visitor becomes a `07_infer` run on the owner's Kaggle account (above), about 12–15 minutes per article, one at a time, at most `DAILY_LIMIT` (default 10) a day. The kaggle CLI signs in from a Kaggle API token in Secret Manager (`kaggle-api-token`, read as `KAGGLE_API_TOKEN`); create it once from a terminal so the token is never typed into a file or chat:
  `read -s T && printf %s "$T" | gcloud secrets create kaggle-api-token --data-file=- --project <gcp-project> && unset T`,
  then let the runtime service account read it (`gcloud secrets add-iam-policy-binding kaggle-api-token --member serviceAccount:<project-number>-compute@developer.gserviceaccount.com --role roles/secretmanager.secretAccessor`). Uploads can be up to 500 MB (`MAX_FILE_MB`): the service runs HTTP/2 end to end (`--use-http2`, served by Hypercorn), because Cloud Run caps HTTP/1 requests at 32 MB. A 51 MB, 47.7-minute upload took 13.1 minutes to an article. YouTube links are refused at once: YouTube blocks downloads from cloud servers. Visitors' runs use the owner's weekly Kaggle GPU hours.
- **CPU** (`Dockerfile.cpu`): 8 vCPUs, 16 GiB, llama.cpp with 4 threads (4 beat 8 by ~20%), Whisper `small`. Measured: an 8.5-minute episode from an RSS feed took 14.5 minutes (95 s transcribing, 13 min writing 2 chunks), roughly 1.5–2× slower than the M1 Air. Episodes are capped at 30 minutes (`MAX_MINUTES`) so a job fits inside Cloud Run's 60-minute request limit. About $0.63 per instance-hour in us-central1.
- **GPU** (`Dockerfile`): needs Cloud Run L4 quota, which Google refused automatically for a new billing account (2026-10-24). The image builds; deploy it once the quota is granted.
