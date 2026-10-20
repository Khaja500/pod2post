# spec.md — Pod2Post: Podcast → SEO Article & Social Content Pipeline (with Post-Training)

> Hand this file to Claude Code. It describes what to build, the constraints (Kaggle free GPUs), how to automate Kaggle runs from the terminal, and how to know when each phase is done.

---

## 1. Goal

Build an automated pipeline that turns a podcast episode (audio or transcript) into:

1. An **SEO-optimized long-form article** (title, meta description, H2/H3 structure, target keywords, FAQ section)
2. **Social content**: LinkedIn post, X/Twitter thread (each post ≤ 280 chars), newsletter blurb, 3–5 pull quotes

The core of the project is **post-training a small open model (~3B parameters)** so it does this cheaply and faithfully, and **showing measurably that post-training helped** versus the base model.

### Headline result we want to report
> "A QLoRA-fine-tuned 3B model, trained with SFT + DPO on Kaggle's free T4 GPUs, produces articles that are X% more faithful to the transcript and score Y on SEO checks, at ~Z× lower cost per article than a large API model."

### Non-goals
- No production web service, auth, or billing.
- No training from scratch or full fine-tuning — QLoRA only.
- No commercial use of research-only datasets (see §4).

---

## 2. Hard constraints (Kaggle free tier)

| Resource | Limit | Design consequence |
|---|---|---|
| GPU | 2× T4 (16 GB each) or 1× P100 | Use **one T4** per run; 4-bit QLoRA; **fp16, not bf16** (T4 has no bf16); no FlashAttention-2 |
| Weekly GPU quota | ~30 h/week (check the quota counter in Kaggle settings; it can change) | Total project budget ≤ 20 GPU-hours; leave headroom |
| Session length | 12 h max for GPU sessions | Every job must finish in < 3 h, checkpoint every N steps, and be resumable |
| Disk | `/kaggle/working` output ~20 GB | Save only LoRA adapters (tens of MB), never merged full weights |
| Secrets | Kaggle Secrets set in the web UI are **not available** to CLI-pushed kernels | Design so Kaggle kernels need **no secrets** (ungated models, no API keys on Kaggle) |
| Internet | Kernels need `enable_internet: true` to pip install; requires a phone-verified Kaggle account | Set in every `kernel-metadata.json` |

**Context-length rule:** A 1-hour episode ≈ 10k words ≈ 13k+ tokens, too long to train on a T4. Therefore the pipeline is **map-reduce**:
- **Stage A (map):** transcript chunk (~1,500 tokens) → structured "episode notes" (key points, claims, quotes, timestamps)
- **Stage B (reduce):** merged notes (≤ 2,500 tokens) → article + social pack

Training max sequence length: **4096 tokens**. Batch size 1, gradient accumulation 8–16, gradient checkpointing on.

---

## 3. Architecture

```
audio (.mp3)
   │  faster-whisper on Kaggle GPU (or skip if transcript exists)
   ▼
transcript (+ timestamps, speakers if available)
   │  clean: remove fillers, fix casing, segment into ~1.5k-token chunks
   ▼
Stage A model:  chunk → episode notes (JSON)
   │  merge + dedupe notes
   ▼
Stage B model:  notes + target keywords → article (Markdown) + social pack (JSON)
   │
   ▼
Scorer: faithfulness, SEO checks, readability, length/format rules
   │
   ▼
outputs/<episode_id>/article.md, social.json, scores.json
```

Stage A and Stage B use the **same base model with one LoRA adapter trained on both tasks** (task indicated in the system prompt). Simpler than two adapters.

---

## 4. Data

### 4.1 Sources (verify licenses before use; record license per source in `data/SOURCES.md`)

| Source | What it gives | License / note | Use |
|---|---|---|---|
| **MediaSum** (github.com/zcgzcgzcg1/MediaSum) | 463.6K NPR/CNN interview transcripts with summaries | **Research use only** | Main source of transcripts for Stage A/B training inputs |
| **SPoRC** (github.com/blitt2018/SPoRC_data) | ~1.1M podcast transcripts (Whisper) from 2020 RSS feeds | **Non-commercial** | Real podcast-style transcripts; sample a few hundred episodes |
| **TalkRL podcast** (HF: `RamAnanth1/talkrl-podcast`) | 39 episodes, Whisper transcripts + summaries + segments | Check card | Small, clean **held-out test set** |
| **Changelog podcast transcripts** (HF: `willtheorangeguy/*-Transcripts`) | Tech podcast transcripts | Check each card's license | Extra test episodes |
| **Creative-Commons audio** (Internet Archive, CC-licensed shows) | Raw audio | CC (check variant) | End-to-end test including Whisper |
| ~~Spotify Podcast Dataset~~ | — | Discontinued Dec 2023, no new access | Do not use |

### 4.2 Building training pairs (distillation)

Real "transcript → SEO article" pairs don't exist publicly, so generate them with a stronger **teacher** model, then filter:

- **Teacher option 1 (default, free):** a larger open model (~14B, 4-bit) run on Kaggle 2×T4 with vLLM or HF generate. Slow but no keys needed on Kaggle.
- **Teacher option 2 (fast, paid):** an API model, run **locally on the user's machine**, never on Kaggle, with the API key read from a local `.env` (git-ignored). Make this configurable in `configs/teacher.yaml`.

Targets:
- 600 Stage A examples (chunk → notes)
- 400 Stage B examples (notes → article + social)
- Hold out **50 episodes** (TalkRL + Changelog) as the test set **before** generating anything. Never train on them.

### 4.3 Filtering (`src/data/filter.py`)
Drop a generated example if any of:
- Article contains a named entity or number not present in the source transcript/notes (entity + number overlap check)
- Fails format rules (missing title/meta/H2s, meta > 160 chars, title > 60 chars, any X post > 280 chars, invalid JSON)
- Near-duplicate of another example (MinHash)

Log the drop rate per rule.

### 4.4 Preference data for DPO (`src/data/build_dpo.py`)
For 300 Stage B inputs, create (chosen, rejected) pairs:
- **chosen** = filtered teacher output
- **rejected** = one of: (a) SFT model output that failed the faithfulness check, (b) teacher output with injected hallucination (fake statistic, invented guest claim), (c) keyword-stuffed version, (d) generic/vague version

Mix ~25% of each rejected type.

---

## 5. Models & training

### 5.1 Base model
- A **~3B instruct model** with an open, **ungated** license and ≥ 8k context. Claude Code: check Hugging Face for the current best ungated ~3B instruct model before starting (examples of the class: Qwen2.5-3B-Instruct or a newer equivalent). Record the choice and why in `docs/DECISIONS.md`.
- Also evaluate the base model zero-shot as the baseline.

### 5.2 Libraries
- **Unsloth** (QLoRA, fp16 on T4) + **TRL** (`SFTTrainer`, `DPOTrainer`)
- **faster-whisper** for ASR
- Pin all versions in `requirements-kaggle.txt`; check current Unsloth Kaggle install instructions before pinning.

### 5.3 SFT config (starting point)
```yaml
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
train_on_responses_only: true
seed: 42
```

### 5.4 DPO config (starting point)
```yaml
start_from: sft_adapter
beta: 0.1
learning_rate: 5e-6
num_train_epochs: 1
max_length: 4096
max_prompt_length: 2560
gradient_accumulation_steps: 8
save_steps: 25
```

### 5.5 Stretch (only if quota remains)
- ORPO single-stage run, compared against SFT+DPO
- GRPO with a programmatic reward (format rules + entity-faithfulness score), 300 steps max

---

## 6. Evaluation (`src/eval/`)

Run every system on the 50 held-out episodes: **base zero-shot**, **SFT**, **SFT+DPO**, and (optional) **teacher** as the upper bound.

| Metric | How |
|---|---|
| **Faithfulness** (primary) | % of article sentences supported by the transcript, via an NLI model (e.g., a DeBERTa MNLI cross-encoder) over sentence ↔ best-matching transcript chunk |
| **Hallucinated entities/numbers** | Count of named entities and numbers in output not found in the transcript |
| **SEO score** | Rule checklist: keyword in title/first 100 words/≥1 H2, title ≤ 60 chars, meta ≤ 160 chars, ≥ 3 H2s, FAQ present, keyword density 0.5–2.5% |
| **Readability** | Flesch reading ease (textstat), target 50–70 |
| **Format validity** | % valid JSON social packs, % X posts ≤ 280 chars |
| **Quality (LLM judge)** | Pairwise judge vs base, run locally with the teacher; report win rate |
| **Cost/latency** | Tokens/sec and seconds per episode on a T4 |

Output: `reports/results.md` with one comparison table and 3 side-by-side example articles.

---

## 7. Kaggle automation (Claude Code must implement this)

### 7.1 Credentials (one-time, user does this)
1. Kaggle → avatar → **Settings → API → Create New Token** → downloads `kaggle.json`
2. Place it at `~/.kaggle/kaggle.json` with `chmod 600` (or set `KAGGLE_USERNAME` / `KAGGLE_KEY` env vars)
3. Verify phone number on Kaggle (needed for GPU + internet)

**Claude Code rules:** never print, log, commit, or copy `kaggle.json` or any key. Add `kaggle.json`, `.env`, `*.key` to `.gitignore` first thing. Verify setup only with `kaggle kernels list --mine`.

### 7.2 Layout: one folder per Kaggle kernel
```
kaggle/
  01_transcribe/   kernel-metadata.json, run.ipynb
  02_teacher_gen/  kernel-metadata.json, run.ipynb   (only if teacher option 1)
  03_sft/          kernel-metadata.json, run.ipynb
  04_dpo/          kernel-metadata.json, run.ipynb
  05_eval/         kernel-metadata.json, run.ipynb
```

Template `kernel-metadata.json`:
```json
{
  "id": "<KAGGLE_USERNAME>/pod2post-03-sft",
  "title": "pod2post-03-sft",
  "code_file": "run.ipynb",
  "language": "python",
  "kernel_type": "notebook",
  "is_private": true,
  "enable_gpu": true,
  "enable_internet": true,
  "dataset_sources": ["<KAGGLE_USERNAME>/pod2post-data"],
  "kernel_sources": [],
  "competition_sources": []
}
```

- Data is uploaded as a **private Kaggle Dataset** (`kaggle datasets create` / `kaggle datasets version`).
- Chain stages by attaching the previous kernel's output via `kernel_sources` (e.g., `04_dpo` lists `<user>/pod2post-03-sft` so the SFT adapter is available as input).
- Notebooks are **generated from `src/` Python modules** by `scripts/build_notebook.py` (wrap code into cells) so all logic lives in version-controlled `.py` files. Never edit notebooks on the Kaggle web UI.

### 7.3 CLI wrapper: `scripts/kaggle_run.py`
```
python scripts/kaggle_run.py check             # verify credentials + network access to Kaggle
python scripts/kaggle_run.py push  03_sft      # build notebook, guard, push (starts a GPU run)
python scripts/kaggle_run.py status 03_sft     # queued / running / complete / error
python scripts/kaggle_run.py wait  03_sft      # poll every 60s until done (timeout 4h)
python scripts/kaggle_run.py pull  03_sft      # kaggle kernels output → runs/<date>/03_sft/
python scripts/kaggle_run.py pipeline          # 01→05 in order, stop on first error
```
Required behavior:
- **Pre-flight guard:** refuse to push if that kernel's status is `running` or `queued` (avoids duplicate sessions burning quota), unless `--force`.
- **Quota log:** append start/end time and duration of every run to `runs/quota_log.csv`; warn if this week's total > 20 h.
- **Resumability:** training notebooks check `/kaggle/input/<prev-run>/checkpoints` and resume from the latest checkpoint if present.
- On error, pull the log and print the last 50 lines.

### 7.4 Running from Claude Code on the web (cloud sessions)
The code must work identically from a local machine and from a Claude Code cloud session:
- Read credentials from `KAGGLE_USERNAME` / `KAGGLE_KEY` env vars if `~/.kaggle/kaggle.json` is absent. Never write them to disk inside the repo.
- The cloud environment must allow outbound access to `kaggle.com` and `www.kaggle.com` (custom network allowlist). `scripts/kaggle_run.py check` must test connectivity with `kaggle kernels list --mine` and print a clear message if blocked.
- All state needed between sessions (quota log, run IDs, phase status) lives in the repo (`runs/quota_log.csv`, `runs/state.json`) and is committed, since cloud sessions start fresh.
- Large artifacts (adapters, generated datasets) are never committed; they live as Kaggle kernel outputs or Kaggle Datasets, referenced by ID in `runs/state.json`.

### 7.5 Smoke tests before any real run
Each kernel accepts `SMOKE=1` (via a config cell): 10 examples, 5 steps, ends in < 10 minutes. Always run the smoke version first after any code change.

---

## 8. Repository structure
```
pod2post/
  spec.md
  README.md
  .gitignore
  requirements.txt             # local
  requirements-kaggle.txt      # pinned for Kaggle
  configs/  sft.yaml dpo.yaml teacher.yaml eval.yaml
  data/     SOURCES.md raw/ processed/ (git-ignored except SOURCES.md)
  src/
    asr/transcribe.py
    data/download.py chunk.py build_sft.py filter.py build_dpo.py
    train/sft.py dpo.py
    infer/pipeline.py          # transcript → notes → article + social
    eval/faithfulness.py seo.py judge.py run_eval.py
  scripts/ build_notebook.py kaggle_run.py
  kaggle/  01_transcribe/ ... 05_eval/
  app/     gradio_app.py       # demo
  reports/ results.md
  docs/    DECISIONS.md
  tests/   test_seo.py test_filter.py test_chunk.py
```

---

## 9. Phases & acceptance criteria

| Phase | Deliverable | Done when | GPU budget |
|---|---|---|---|
| 0. Setup | Repo, .gitignore, Kaggle CLI verified, `kaggle_run.py` with smoke kernel | `kaggle_run.py pipeline` works end-to-end in SMOKE mode | < 0.5 h |
| 1. Data | Downloads, held-out test split, chunking, SOURCES.md | 50 test episodes frozen; chunker unit tests pass | 0 h |
| 2. Transcribe | Whisper on 5 CC audio episodes | Transcripts saved; WER spot-checked on 1 episode | ~1 h |
| 3. Teacher gen | 600 Stage A + 400 Stage B examples, filtered | Drop-rate report; ≥ 70% of target count survives | 0–6 h (option 1) |
| 4. Baseline eval | Base model zero-shot on test set | `results.md` has baseline row | ~1 h |
| 5. SFT | LoRA adapter | Beats baseline on faithfulness and SEO score | ~2–3 h |
| 6. DPO | DPO adapter | Fewer hallucinated entities than SFT, no SEO regression | ~2 h |
| 7. Eval + report | Full table, examples, cost/latency | `reports/results.md` complete | ~2 h |
| 8. Demo | Gradio app | Paste transcript or upload short audio → article + social | 0 h |

Budget estimates are rough; measure after the smoke runs and update this table.

---

## 10. Demo / where to test

- **Primary:** `05_eval` Kaggle kernel generates outputs for all test episodes; review them in `runs/`.
- **Interactive demo:** `app/gradio_app.py` loading the base model + LoRA adapter. Run it inside a Kaggle notebook (GPU) or locally. Optionally publish to a Hugging Face Space using a quantized (GGUF) version of the merged model for CPU inference; note it will be slow.
- **Real-world test:** run the full pipeline on 2–3 Creative-Commons podcast episodes not in any dataset.

---

## 11. Rules for Claude Code

1. Read this spec fully; ask before changing scope or the base model.
2. Check current versions of Unsloth, TRL, and faster-whisper, plus the Kaggle install method, before writing code.
3. Secrets: never print, log, or commit credentials. Kaggle kernels must not need any secret.
4. Always run SMOKE mode before a full Kaggle run; never start a full run while another is running.
5. Never train on the held-out test set; assert this in code (hash test episode IDs, check at train time).
6. Record every non-obvious choice in `docs/DECISIONS.md`.
7. After each phase, update the phase table status and `runs/quota_log.csv`, and stop for user review before starting the next GPU-heavy phase.
