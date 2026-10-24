# spec.md — Pod2Post: Podcast → SEO Article & Social Content Pipeline (with Post-Training)

> Hand this file to Claude Code. It describes what to build, the constraints (Kaggle free GPUs), how to automate Kaggle runs from the terminal, and how to know when each phase is done.
>
> **Revised 2026-10-22** to match what was built and measured. Every change has its reason in `docs/DECISIONS.md` (D1–D33). The main ones:
> - **Models:** the base is Qwen3-4B-Instruct-2507 (4B) and the teacher is Qwen3-14B-AWQ on vLLM.
> - **Prompt sizing:** merged notes are capped at 1,800 tokens and articles run 700–1,000 words, so every training example fits 4,096 tokens.
> - **Test set:** 38 TalkRL + 12 Changelog episodes.
> - **Kaggle tooling:** Kaggle CLI 2.x auth, and the cloud allowlist includes `api.kaggle.com`.
> - **Automation:** one-command `pipeline` with automatic smoke runs and resume.
> - **Cost:** a free cost estimate replaces a paid API comparison.
> - **Judging:** an LLM-judge kernel (`06_judge`).

---

## 1. Goal

Build an automated pipeline that turns a podcast episode (audio or transcript) into:

1. An **SEO-optimized long-form article** (title, meta description, H2/H3 structure, target keywords, FAQ section)
2. **Social content**: LinkedIn post, X/Twitter thread (each post ≤ 280 chars), newsletter blurb, 3–5 pull quotes

The core of the project is **post-training a small open model (~3–4B parameters)** so it does this cheaply and faithfully, and **showing measurably that post-training helped** versus the base model.

### Headline result we want to report
> "A QLoRA-fine-tuned 4B model, trained with SFT + DPO on Kaggle's free T4 GPUs, produces articles that are X% more faithful to the transcript and score Y on SEO checks, at ~Z× lower cost per article than a large API model."

`src/eval/report.py` writes this sentence from measured numbers, and states a regression plainly if there is one.
- **X and Y** come from the evaluation (§6).
- **Z** is a free estimate (§6, "Cost"): the pipeline's measured token counts × published API prices, against one T4 at its on-demand price. No API is called.

### Non-goals
- No production web service, auth, or billing.
- No training from scratch or full fine-tuning — QLoRA only.
- No commercial use of research-only datasets (see §4).
- No paid API calls. The cost comparison is an estimate (§6).

---

## 2. Hard constraints (Kaggle free tier)

| Resource | Limit | Design consequence |
|---|---|---|
| GPU | `NvidiaTeslaT4` = a 2× T4 machine (16 GB each); P100 is no longer offered | Training uses **one T4** (`CUDA_VISIBLE_DEVICES=0` in code); the teacher, eval generation and judge use both (tensor parallel). 4-bit QLoRA; **fp16, not bf16** (T4 = sm_75, verified); no FlashAttention-2 |
| Weekly GPU quota | 30 h/week (`kaggle quota`; resets Saturdays 00:00 UTC) | Total project budget ≤ 30 GPU-hours (raised from 20 on 2026-10-22); leave headroom |
| Session length | 12 h max; we cap every run at **3 h** (`kaggle kernels push -t`) | Long jobs stop themselves at a time budget (teacher 2 h 15 m, SFT 2 h 30 m, DPO 2 h), save, and are **resumed automatically** |
| Disk | `/kaggle/working` output ~20 GB | Save only LoRA adapters, never merged weights. Caches (HF, Unsloth) go to `/tmp` |
| Secrets | Kernels pushed from the CLI get no Kaggle Secrets | Kernels need **no secrets**: ungated models only, no API keys on Kaggle |
| Internet | `enable_internet: true`; phone-verified account | Set in every `kernel-metadata.json` |

**Context-length rule:** A 1-hour episode ≈ 10k words ≈ 13k+ tokens, too long to train on a T4. Therefore the pipeline is **map-reduce**:
- **Stage A (map):** transcript chunk (≤ 1,500 tokens, ~150 overlap) → structured "episode notes" (JSON: summary, key points, claims, quotes, entities, numbers, keywords, timestamps)
- **Sanitise (deterministic):** drop notes quotes that aren't verbatim in their chunk, and entities or figures the chunk doesn't contain
- **Stage B (reduce):** merged notes (**≤ 1,800 tokens**) + target keywords → article (Markdown, 700–1,000 words) + social pack (JSON)

Training max sequence length: **4096 tokens**. A Stage B example (prompt ≈ 2.3k + response ≈ 1.6k) fits, and longer examples are dropped and logged. Batch size 1, gradient accumulation 8, gradient checkpointing on.

---

## 3. Architecture

```
audio (.mp3)
   │  faster-whisper large-v3-turbo on a Kaggle T4 (or skip if a transcript exists)
   ▼
transcript (+ timestamps, speakers if available)
   │  clean: fillers, stutters, markdown escapes, stage directions; merge ASR segments into sentences;
   │  segment into ≤ 1.5k-token chunks with ~150-token overlap (base-model tokenizer)
   ▼
Stage A model:  chunk → episode notes (JSON)      [one low-temperature retry on invalid JSON]
   │  sanitise against the chunk, merge + dedupe (≤ 1,800 tokens), pick primary + 3 secondary keywords
   ▼
Stage B model:  notes + target keywords → article (Markdown) + social pack (JSON)
   │
   ▼
Scorer: faithfulness (NLI), invented entities/numbers, SEO checklist, readability, format rules
   │
   ▼
outputs/<system>.jsonl, scores/<system>.jsonl, results.json  →  reports/results.md
```

Stage A and Stage B use the **same base model with one LoRA adapter trained on both tasks** (task indicated in the system prompt). The teacher, the student and the evaluation use the **same prompts** (`src/data/prompts.py`) and the **same pipeline** (`src/infer/pipeline.py`).

---

## 4. Data

### 4.1 Sources (licenses verified 2026-10-20; details in `data/SOURCES.md`)

| Source | What it gives | License / note | Use |
|---|---|---|---|
| **MediaSum** (`ccdv/mediasum`, validation zip) | NPR/CNN interview transcripts | **Research use only** | **Training**: 2,000 interviews, 1,775 sampled at 600–6k words plus all 225 at 6k–25k words. The long ones are kept because test episodes have a median of 11 chunks |
| **TalkRL podcast** (HF `RamAnanth1/talkrl-podcast`) | 39 rows of Whisper transcripts + summaries + timestamps | **No license stated**: evaluation only, never redistributed | **Test**: 38 episodes (the 305-word trailer is excluded) |
| **The Changelog** (github.com/thechangelog/transcripts) | Human-edited tech podcast transcripts | **CC BY-SA 4.0** (the HF copies are derived, so this applies) | **Test**: 12 episodes, #662–673 |
| **Creative-Commons audio** (Internet Archive) | Raw audio | CC BY / BY-SA (per show) | End-to-end Whisper test: 5 episodes (Going Linux, Open Source, Libre Lounge, TCLP, mintCast), ~4.8 h |
| Changelog #661 audio | Audio with a human transcript | Audio © Changelog Media: used privately, for the WER check only | WER reference |
| ~~SPoRC~~ | Podcast transcripts | Gated on HF (needs a login) | **Not used** |
| ~~Spotify Podcast Dataset~~ | — | Discontinued Dec 2023 | Do not use |

### 4.2 Building training pairs (distillation)

Real "transcript → SEO article" pairs don't exist publicly, so a stronger **teacher** generates them and they are then filtered:

- **Teacher option 1 (chosen, free): `Qwen/Qwen3-14B-AWQ` on vLLM 0.32**, tensor-parallel across Kaggle's 2× T4, with thinking disabled. This is kernel `02_teacher_gen`. It is resumable and has a time budget. A Stage B output that breaks a hard format rule is retried once, for data generation only. It also writes 100 "generic" rewrites for DPO (§4.4).
- **Teacher option 2 (fast, paid):** an API model, run **locally only**, with the key in `.env` (git-ignored). Not used.

Targets:
- 600 Stage A examples (chunk → notes)
- 400 Stage B examples (notes → article + social). The teacher generates 30% extra (520 episodes, 25% of them long) to allow for filtering.
- **50 held-out episodes** (§4.1) were frozen **before** any generation, and are stored as SHA-256 hashes in `data/processed/test_ids.sha256`. Every training entry point asserts no overlap.

### 4.3 Filtering (`src/data/filter.py`)
Drop a generated example if any of:
- **An entity or number in the output is not in the source.** For Stage B the source is the **transcript only**, so entities invented in the notes can't reach training data.
  - Entities come from spaCy NER on prose and social text. Headings and hashtags are skipped.
  - An entity is grounded if it appears literally, or as a normal variant: aliases and demonyms (U.S./America/American), plurals, acronyms in either direction (ACA ↔ Affordable Care Act), or a full name whose surname is in the source.
  - Numbers above 10 must appear in the source.
- **Stage A only:** a quote is not verbatim in the chunk.
- **Format rules fail:** missing title/meta/≥3 H2s/FAQ, meta > 160 chars, title > 60 chars, any X post > 280 chars, invalid JSON. Teacher JSON gets a narrow repair first; evaluation is strict.
- **Near-duplicate** of another example (MinHash over 5-grams, threshold 0.8).

The drop rate per rule is logged to `filter_report.json`.

### 4.4 Preference data for DPO (`src/data/build_dpo.py`, kernel `04_dpo`)
For the 300 Stage B training inputs, create (chosen, rejected) pairs (D39):
- **(a) on-policy, the main signal.** `04_dpo` samples the SFT model twice per prompt (vLLM on both T4s, in its own venv, until a 50-minute budget), then scores every sample and the teacher's answer with the eval's own scorer (NLI faithfulness, invented entities/numbers, SEO, format). **rejected** = the least faithful sample; **chosen** = the most faithful answer that passes every check, from the teacher or the other sample. A pair is kept only if chosen is ≥ 8 points more faithful, scores no lower on SEO, and is at least 85% as long (DPO otherwise learns "shorter is better").
- **(b)–(d) synthetic, a third of the pairs**, round-robin on the prompts without an (a) pair. chosen = filtered teacher output; rejected =
  - (b) teacher output with an injected hallucination: a fake statistic plus an invented claim attributed to a real speaker.
  - (c) a keyword-stuffed version, with density well above 2.5%.
  - (d) a generic, vague rewrite written by the teacher.

The first DPO run (D35) used (a) = "an SFT output that fails the grounding or format checks", mixed ~25% each with (b)–(d); 265 of its 285 pairs were synthetic and it changed no metric (D39).

The DPO reference model is the **SFT adapter**, kept as a frozen copy, not the base model.

---

## 5. Models & training

### 5.1 Base model
- **`Qwen/Qwen3-4B-Instruct-2507`**, trained from `unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit`.
  - It is dense, non-thinking, Apache-2.0, ungated, with 256k context and first-class Unsloth T4 support. It has 4.0B parameters, slightly above "~3B".
  - Rejected: Qwen3.5-4B (hybrid linear attention, a T4 risk) and Granite-4.2-3B (a reasoning model, no Unsloth build). The fallback is SmolLM3-3B. See D16.
- The base model is also evaluated zero-shot as the baseline.

### 5.2 Libraries (pinned in `requirements-kaggle.txt`, verified on Kaggle 2026-10-20)
- **Unsloth 2026.10.7** with transformers 5.17.0, TRL 1.15.0, PEFT 0.21.2, bitsandbytes 0.50.2 and datasets 4.8.5. torch stays at the Kaggle image's version (2.11).
- **vLLM 0.32.0** for the teacher, evaluation and judge. It pins its own torch (2.13), so each kernel installs only what it needs, and kernels that install vLLM remove the clashing torchaudio.
- **faster-whisper 1.2.1** with **PyAV 18.1.0**, because PyAV 19 broke faster-whisper.

### 5.3 SFT config (`configs/sft.yaml`)
```yaml
base_model: unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit
load_in_4bit: true
dtype: float16
max_seq_length: 4096
lora_r: 16
lora_alpha: 32
lora_dropout: 0
target_modules: [q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]
per_device_train_batch_size: 1
gradient_accumulation_steps: 8
learning_rate: 2e-4
num_train_epochs: 2
warmup_ratio: 0.05
lr_scheduler: cosine
save_steps: 50
train_on_responses_only: true   # done with TRL's prompt-completion format + completion_only_loss
seed: 42
time_budget_s: 9000             # save + stop at 2h30m; resumed automatically
```

### 5.4 DPO config (`configs/dpo.yaml`)
```yaml
start_from: sft_adapter         # TRL keeps a frozen copy as the reference
beta: 0.1
learning_rate: 1e-5             # 5e-6 in the first run, which barely moved the adapter (D39)
num_train_epochs: 1
max_length: 4096                # max_prompt_length was removed in TRL 1.x; prompts fit by construction
gradient_accumulation_steps: 8
save_steps: 25
sample_time_budget_s: 3000      # SFT sampling on both T4s
faith_margin: 0.08              # (a) pairs: chosen >= 8 points more faithful
synthetic_share: 0.33           # (b)-(d) pairs as a third of the total
time_budget_s: 5400             # at most 1.5 h of training,
kernel_budget_s: 9900           # and never past 2h45m from kernel start
```

### 5.5 Stretch (only if quota remains)
- ORPO single-stage run, compared against SFT+DPO
- GRPO with a programmatic reward (format rules + entity-faithfulness score), 300 steps max

---

## 6. Evaluation (`src/eval/`, kernels `05_eval` and `06_judge`)

Run every system on the 50 held-out episodes with the same pipeline, prompts and sampling (temperature 0.3): **base zero-shot**, **SFT**, **SFT+DPO**. Optionally run the **teacher** as an upper bound. SFT+DPO writes its articles from **SFT's Stage A notes**: DPO was trained on Stage B only, and sharing the notes isolates the comparison to the article step (D38). The run is resumable: 8 episodes per batch, saved after each, no new batch after 2 h, resumed automatically until every system is done (D37).

| Metric | How |
|---|---|
| **Faithfulness** (primary) | % of article prose sentences supported by the transcript. Each sentence → its top-3 BM25 transcript windows → NLI (`MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli`), supported if p(entailment) ≥ 0.5 |
| **Hallucinated entities/numbers** | Entities and numbers in the output not grounded in the transcript (same function as the filter, §4.3) |
| **SEO score** | Rule checklist: keyword in title / first 100 words / ≥1 H2, title ≤ 60 chars, meta ≤ 160 chars, ≥ 3 H2s, FAQ present, keyword density 0.5–2.5% (occurrences / words) |
| **Readability** | Flesch reading ease (textstat) on prose sentences only, target 50–70 |
| **Format validity** | % valid JSON social packs (strict, no repair), % X posts ≤ 280 chars |
| **Quality (LLM judge)** | `06_judge`: the teacher compares each system with base, in **both A/B orders**, using transcript passages retrieved for both articles. Reports win/tie/loss |
| **Cost/latency** | Tokens/sec and seconds per episode on **two** T4s (vLLM tensor parallel, 8 episodes per batch); T4-seconds = wall time × 2. SFT+DPO's times cover Stage B only |
| **Cost per article** (free estimate) | `configs/costs.yaml` + `src/eval/cost.py`. API cost = the pipeline's prompt tokens (rebuilt exactly) + measured completion tokens × published per-token prices (OpenRouter, date recorded). Self-hosted cost = T4-seconds per episode × one T4's $/hour (AWS g4dn.xlarge on-demand as reference; Kaggle free), measured on the system that ran the whole pipeline (SFT, whose notes SFT+DPO reuses). The one-time training GPU-hours are reported separately. **No API is called** |

Output: `reports/results.md` with the comparison table, SEO check breakdown, cost table, judge table, 3 side-by-side example articles and the headline sentence (§1).

---

## 7. Kaggle automation

### 7.1 Credentials (one-time, user does this)
1. Kaggle CLI 2.x: run `kaggle auth login` (OAuth), or save a token from Settings → API → **Generate New Token** to `~/.kaggle/access_token` (`chmod 600`). The legacy `kaggle.json` / `KAGGLE_USERNAME` + `KAGGLE_KEY` still work.
2. Verify the phone number on Kaggle, which GPU and internet access need.

**Claude Code rules:**
- Never print, log, commit or copy any credential. `.gitignore` covers `kaggle.json`, `access_token`, `credentials*.json`, `.env` and `*.key`.
- `scripts/kaggle_run.py` only checks that a credential source exists, and shells out to the `kaggle` CLI.
- Verify setup with `kaggle_run.py check`.

### 7.2 Layout: one folder per Kaggle kernel
```
kaggle/
  00_smoke/        environment check (GPU, fp16, pinned installs)
  01_transcribe/   kernel-metadata.json, main.py
  02_teacher_gen/  …
  03_sft/  04_dpo/  05_eval/  06_judge/
```
- Each folder has a `kernel-metadata.json` **template** (`<KAGGLE_USERNAME>` placeholder) and `main.py` (percent-format cells).
- `scripts/build_notebook.py` renders `build/kaggle/<slug>/run.ipynb` (git-ignored) and bundles `src/`, `configs/` and `requirements-kaggle.txt` into the notebook. Notebooks are never edited on Kaggle.
- `machine_shape: "NvidiaTeslaT4"`, private, GPU and internet on.
- **Smoke runs push to separate `<slug>-smoke` kernels and read a separate `pod2post-data-smoke` dataset**, so they never touch full-run outputs or data.
- Data lives in private Kaggle Datasets (`pod2post-data`, `pod2post-audio`). Stages chain through `kernel_sources` (e.g. `04_dpo` attaches `03_sft`'s output).

### 7.3 CLI wrapper: `scripts/kaggle_run.py`
```
python scripts/kaggle_run.py check                  # CLI, credentials, network (per host), quota
python scripts/kaggle_run.py push 03_sft [--full]   # build notebook, pre-flight guard, push
python scripts/kaggle_run.py status|wait|pull|logs 03_sft [--full]
python scripts/kaggle_run.py dataset [data|audio]   # create/version a private dataset (skipped if unchanged)
python scripts/kaggle_run.py build-data --full      # filter + build SFT/DPO data + upload, from the latest 02 output
python scripts/kaggle_run.py pipeline [--full] [--from 03_sft]   # the whole project, 01 → 06
python scripts/kaggle_run.py quota
```
Required behavior:
- **Pre-flight guard.** `push` refuses, unless `--force`, in any of these cases:
  - the kernel is queued or running
  - an upstream kernel or dataset isn't ready
  - `--full` is requested without a completed smoke run of the **same code hash**. The hash covers only the files the kernel imports or launches.
  - `--full` is requested while another run is active
- **Pipeline:** smoke-checks each stage first when its code changed, then runs the local steps after 02, resumes on the time budget, and computes WER (01) and the report (05/06).
- **Quota log:** `runs/quota_log.csv` records in-kernel runtime (or wall time) per run, with a warning above 20 h/week. `kaggle quota` is authoritative.
- **Resumability:** `--resume` attaches the kernel's own previous output, and training continues from the newest checkpoint while the teacher skips finished records.
- **On error:** pull the log and print the last 50 lines.
- **Robustness:** read-only calls retry network outages, downloads retry connection resets, and a transient "absent" status is retried. Pushes are never retried.

### 7.4 Running from Claude Code on the web (cloud sessions)
- Credentials come from `KAGGLE_API_TOKEN` (or `KAGGLE_USERNAME` + `KAGGLE_KEY`). They are never written to the repo.
- The network allowlist must include **`api.kaggle.com`** (Kaggle CLI 2.x), `www.kaggle.com`, `kaggle.com`, and `storage.googleapis.com` (file transfers). `check` probes each one.
- The state needed between sessions lives in the repo and is committed: `runs/quota_log.csv` and `runs/state.json`.
- Large artifacts are never committed.

### 7.5 Smoke tests before any real run
Each kernel runs in SMOKE mode (`SMOKE=1` in the generated config cell) on 10 examples or 2–3 episodes, for 5 steps. Kernels that install vLLM (02, 04, 05, 06) get 50 minutes for install, model download and engine start. `pipeline --full` runs a stage's smoke version automatically whenever the stage's code changed.

---

## 8. Repository structure
```
pod2post/
  spec.md plan.md README.md .gitignore
  requirements.txt             # local (.venv)
  requirements-kaggle.txt      # pinned for Kaggle
  configs/  data.yaml asr.yaml teacher.yaml sft.yaml dpo.yaml eval.yaml costs.yaml
  data/     SOURCES.md processed/test_ids.sha256 (rest git-ignored)
  src/
    kaggle_env.py              # kernel runtime helpers
    asr/   transcribe.py wer.py
    data/  download.py chunk.py testset.py prompts.py notes.py teacher.py filter.py build_sft.py build_dpo.py
    train/ sft.py dpo.py sample_sft.py
    infer/ generator.py pipeline.py
    eval/  seo.py faithfulness.py run_eval.py kernel.py judge.py cost.py report.py
  scripts/ build_notebook.py kaggle_run.py
  kaggle/  00_smoke/ 01_transcribe/ 02_teacher_gen/ 03_sft/ 04_dpo/ 05_eval/ 06_judge/
  app/     gradio_app.py
  reports/ results.md
  docs/    DECISIONS.md
  runs/    state.json quota_log.csv (+ pulled outputs, git-ignored)
  tests/   test_chunk.py test_filter.py test_seo.py … (84 tests)
```

---

## 9. Phases & acceptance criteria

| Phase | Deliverable | Done when | GPU budget | Status (2026-10-24) |
|---|---|---|---|---|
| 0. Setup | Repo, .gitignore, Kaggle CLI verified, `kaggle_run.py`, smoke kernel | `pipeline` works end-to-end in SMOKE mode | < 0.5 h | ✅ Done. Smoke pipeline green 01 → 05 |
| 1. Data | Downloads, held-out test split, chunking, SOURCES.md | 50 test episodes frozen; chunker unit tests pass | 0 h | ✅ Done |
| 2. Transcribe | Whisper on 5 CC audio episodes | Transcripts saved; WER spot-checked on 1 episode | ~1 h | ✅ Done. 0.2 h. WER 21.1% raw, 4.1% substitutions + deletions (D33) |
| 3. Teacher gen | 600 Stage A + 400 Stage B examples, filtered | Drop-rate report; ≥ 70% of target survives | 0–6 h | ✅ Done. 7.1 h incl. smoke. Kept 1,331 A / 315 B (D31) |
| 4. Baseline eval | Base model zero-shot on the test set | `results.md` has the baseline row | ~1 h | ✅ Done in `05_eval` (2026-10-23): faithfulness 43.5%, SEO 80.5%, Flesch 48.7 |
| 5. SFT | LoRA adapter | Beats baseline on faithfulness and SEO score | ~2–3 h | ❌ Trained (2 epochs, val loss 0.370 → 0.345, 2.8 h) but **fails**: SEO 97.8% (base 80.5%), faithfulness 38.6% (base 43.5%; 95% CI of the gap −8.5 to −1.3 points), Flesch 29.7. Fewer articles invent anything (26% vs 44%) |
| 6. DPO | DPO adapter | Fewer hallucinated entities than SFT, no SEO regression | ~2 h | ❌ **Fails** after 3 runs. Run 1 (285 pairs, 93% synthetic, D35) and run 3 (70 on-policy scored pairs, 3 epochs, D39–D40) are both within noise of SFT: run 3 has 0.42 vs 0.42 invented entities, faithfulness 39.1% vs 38.6% (CI of the gap −2.9 to +3.9), SEO 96.5% vs 97.8%. On MediaSum, SFT is already almost as faithful as the teacher (51.2% vs 53.8%) |
| 7. Eval + report | Full table, examples, cost/latency, judge | `reports/results.md` complete | ~6 h | ✅ Done (2026-10-23). Full eval 3.6 h + re-evals 1.2 h (D38–D41); judge 1.05 h: base beats SFT 48% / SFT + DPO 52% of episodes, the systems win 12% / 14%. Ablation (D41): with base's notes, SFT's writer is 6.4 points less faithful than base's (CI −10.6 to −2.5), so the loss is in the article style |
| 8. Demo | Gradio website | Podcast link, audio file or transcript → article + social | 0 h | ✅ Built and public (`app/gradio_app.py`, 2026-10-23/24): direct audio links, RSS feeds and episode pages (yt-dlp) via `src/asr/url_audio.py`, faster-whisper transcription, `.md` download. Runs on a Mac (llama.cpp, 4-bit GGUF + adapter; 48-min episode in 47 min), and on Kaggle's GPUs with the trained adapters (`07_infer`, D43; 48-min episode uploaded to the public site → article in 12.3 min, D44). Public on Cloud Run as a front end that sends every job to Kaggle (D42–D44) |

GPU used: 27.9 h of the 30 h budget (Kaggle's counter, 2026-10-23). Building the models is finished; the 2.1 h left cannot pay for new teacher targets and an SFT retrain (~10 h), the likely fix (D41). Since 2026-10-24 the website writes articles on Kaggle (`07_infer`, D43); those runs are inference from the weekly quota, outside the 30 h build budget, and are logged in `runs/quota_log.csv`.

---

## 10. Demo / where to test

- **Primary:** the `05_eval` kernel generates outputs for all test episodes, and `06_judge` compares them. Review them in `runs/` and `reports/results.md`.
- **Interactive demo:** `app/gradio_app.py` loads the base model + LoRA adapter and runs the same pipeline that was evaluated (`src/infer/article.py`).
  - Public: the Cloud Run site (`deploy/cloudrun/deploy.sh`, `MODE=kaggle`); every article runs on the owner's Kaggle GPUs (D43).
  - Mac: `python3 app/gradio_app.py` (llama.cpp; add `--kaggle` for the Kaggle GPU models).
  - Kaggle notebook (GPU, vLLM): `--adapter /kaggle/input/pod2post-04-dpo/adapter --share`.
  - Batch, from a terminal: `python3 scripts/kaggle_infer.py run --url <link> ...`.
  - Locally: `--backend hf` (transformers, CPU or Apple GPU; slow).
  - Optionally publish to a Hugging Face Space with a GGUF of the merged model.
- **Real-world test:** the full pipeline runs on the CC episodes transcribed in Phase 2, which are not in any dataset.

---

## 11. Rules for Claude Code

1. Read this spec fully. Ask before changing scope; base-model and teacher choices are recorded in DECISIONS.
2. Check current versions of Unsloth, TRL, vLLM and faster-whisper, plus the Kaggle install method, before writing code.
3. Secrets: never print, log, or commit credentials. Kaggle kernels must not need any secret.
4. Always run SMOKE mode before a full Kaggle run (`pipeline --full` enforces this), and never start a full run while another is running.
5. Never train on the held-out test set. This is asserted in code: test episode IDs are hashed and checked at train time.
6. Record every non-obvious choice in `docs/DECISIONS.md`.
7. After each phase, update the phase table status and `runs/quota_log.csv`. The user has since asked for the phases to run back to back without stopping for review. Failures and scope or budget changes still stop the pipeline.
