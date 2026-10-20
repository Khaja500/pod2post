# plan.md — Pod2Post Execution Plan

This is the step-by-step order of work. `spec.md` says **what** to build; this file says the **order**, **where** (local vs cloud), and **what to type into Claude Code** at each step.

Legend: 🧑 = you do it · 🤖 = Claude Code does it · 💻 = run Claude Code locally · ☁️ = run Claude Code on the web (claude.ai/code)

Total Kaggle GPU budget: ≤ 20 GPU-hours.

---

## Part 0 — One-time setup (all 🧑)

- [ ] **Kaggle account:** sign up, then **verify your phone number** (required for GPU + internet in notebooks).
- [ ] **Kaggle API token:** kaggle.com → avatar → Settings → API → **Create New Token** → downloads `kaggle.json`.
  - Local: move it to `~/.kaggle/kaggle.json`, then `chmod 600 ~/.kaggle/kaggle.json` (Windows: `C:\Users\<you>\.kaggle\kaggle.json`).
- [ ] **Kaggle CLI:** `pip install kaggle`, then test with `kaggle kernels list --mine` (an empty list with no error = working).
- [ ] **GitHub repo:** create a private repo `pod2post`, clone it, put `spec.md` and `plan.md` in it, commit, push.
- [ ] **Claude Code locally:** install it and open it in the repo folder.
- [ ] **Decide the teacher option** (spec §4.2):
  - Option 1: free open model on Kaggle (~6 extra GPU-hours, slower)
  - Option 2: paid API model run on your machine (fast; a few dollars for ~1,000 examples)

---

## Part 1 — Foundations (💻 local)

### Step 1 · Phase 0: Repo scaffold + Kaggle automation
🤖 **Prompt:**
> Read spec.md and plan.md fully. Do Phase 0 only: create the repo structure from spec §8, the .gitignore (secrets first), requirements files, `scripts/build_notebook.py`, and `scripts/kaggle_run.py` with check/push/status/wait/pull/pipeline commands and the pre-flight guard and quota log. Create a minimal `kaggle/00_smoke` kernel that prints GPU info and writes a file to /kaggle/working. Run `kaggle_run.py check`, then push, wait, and pull the smoke kernel. Stop and show me the result.

✅ **Done when:** smoke kernel runs on a T4, output is pulled into `runs/`, `quota_log.csv` has one row.
🧑 **Check:** open the kernel on kaggle.com and confirm it says "GPU T4" and is private.

### Step 2 · Phase 1: Data
🤖 **Prompt:**
> Do Phase 1 from spec.md. First check current licenses for MediaSum, SPoRC, TalkRL, and the Changelog transcript datasets and record them in data/SOURCES.md. Download samples (MediaSum: 2,000 interviews; SPoRC: 500 episodes; all of TalkRL). Freeze the 50-episode held-out test set and save its hashed IDs. Implement and unit-test the chunker (~1,500-token chunks with overlap). Upload processed data as a private Kaggle Dataset `pod2post-data`. Stop and summarize dataset sizes.

✅ **Done when:** `tests/test_chunk.py` passes, the test set is frozen, the Kaggle Dataset exists.
🧑 **Check:** read 2–3 transcripts yourself. Are they readable? Is the test set made of real podcast episodes?

### Step 3 · Phase 2: Transcription (optional but good for the demo)
🧑 Download 5 Creative-Commons podcast episodes (Internet Archive) and put them in `data/raw/audio/`.
🤖 **Prompt:**
> Do Phase 2: faster-whisper transcription kernel on Kaggle. SMOKE run first on 2 minutes of audio, then the 5 episodes. Report real-time factor and GPU minutes used.

✅ **Done when:** 5 transcripts saved; you've spot-checked one against the audio.

---

## Part 2 — Data generation + training (💻 then ☁️)

### Step 4 · Phase 3: Teacher data generation
🤖 **Prompt (option 1, Kaggle teacher):**
> Do Phase 3 with teacher option 1. Write the Stage A and Stage B prompts, generate in SMOKE mode (10 examples) and show me 3 examples of each before the full run. After I approve, run the full generation, then filtering (spec §4.3). Report the drop rate per rule.

🤖 **Prompt (option 2, API teacher, local only):**
> Do Phase 3 with teacher option 2, running locally. Read the API key from .env (git-ignored). Generate 10 examples first and show me 3 of each before generating the rest. Then filter and report drop rates.

🧑 **Important review:** read ~10 generated articles. If they're bad, the trained model will be bad. Fix the prompts now, not later.
✅ **Done when:** ≥ 420 Stage A + ≥ 280 Stage B examples survive filtering.

### Step 5 · Phase 4: Baseline eval
🤖 **Prompt:**
> Do Phase 4: implement the eval suite (spec §6) and its unit tests, then run the base model zero-shot on the test set on Kaggle. Write the baseline row in reports/results.md.

✅ **Done when:** baseline numbers exist for every metric. This is what you have to beat.

### ☁️ Switch to the cloud here (optional)
🧑 One-time cloud setup at claude.ai/code:
- [ ] Connect GitHub and select the `pod2post` repo.
- [ ] Create an environment: network access = custom, add `kaggle.com` and `www.kaggle.com` (plus the default package registries); environment variables `KAGGLE_USERNAME=...` and `KAGGLE_KEY=...` (from your kaggle.json).
- [ ] First cloud prompt: *"Run `scripts/kaggle_run.py check` and tell me if Kaggle is reachable."*

Note: environment variables are visible to anyone who can edit that environment. Fine for a personal account; never share the environment.

### Step 6 · Phase 5: SFT
🤖 **Prompt:**
> Do Phase 5. Run SFT in SMOKE mode, show me the loss curve, then run the full SFT on Kaggle with checkpointing. When done, run eval on the SFT adapter and add the SFT row to results.md. Update runs/state.json and commit.

✅ **Done when:** SFT beats baseline on faithfulness and SEO score.
❌ **If it doesn't:** check data quality first (Step 4), then learning rate / epochs. Don't just train longer.

---

## Part 3 — DPO, report, demo (☁️ or 💻)

### Step 7 · Phase 6: DPO
🤖 **Prompt:**
> Do Phase 6: build the DPO preference set (spec §4.4) using the SFT adapter's failures plus synthetic rejected answers, then run DPO from the SFT adapter (SMOKE first). Evaluate and add the SFT+DPO row to results.md.

✅ **Done when:** fewer hallucinated entities than SFT and no drop in SEO score.

### Step 8 · Phase 7: Final eval + report
🤖 **Prompt:**
> Do Phase 7: full eval of base, SFT, and SFT+DPO (plus the teacher as upper bound if budget allows). Add the LLM-judge win rates, cost/latency per episode, and 3 side-by-side example articles to reports/results.md. Write the headline result sentence from spec §1 with real numbers.

### Step 9 · Phase 8: Demo
🤖 **Prompt:**
> Do Phase 8: Gradio app per spec §10. It should accept a pasted transcript or a short audio file and return the article and social pack, plus the faithfulness and SEO scores. Add run instructions to the README for both Kaggle and local.

🧑 **Test it** on 2–3 CC episodes that aren't in any dataset. Screenshot good outputs for your README/portfolio.

### Step 10 · Stretch (only if quota remains)
- ORPO vs SFT+DPO comparison
- GRPO with a format + faithfulness reward (≤ 300 steps)

---

## GPU quota budget

| Work | GPU-hours (rough) |
|---|---|
| Smoke tests, transcription | ~1.5 |
| Teacher gen (option 1 only), baseline, SFT | ~4–10 |
| DPO, final eval | ~4 |
| **Total** | **~10–16** |

Check the quota counter in your Kaggle settings before starting a big run.

---

## Rules of thumb while working with Claude Code

1. **One phase per prompt.** Ask it to stop after each phase so you can review.
2. **Always SMOKE first.** A broken full run wastes hours of quota.
3. **Look at outputs, not just metrics.** Read generated articles at every phase.
4. **Never paste your Kaggle key into the chat.** It belongs in `~/.kaggle/kaggle.json` (local) or environment variables (cloud).
5. **If a session ends mid-run,** start a new one with: *"Read runs/state.json and plan.md, tell me where we are, and continue from the last completed step."*

---

## What to put on your resume when done
> Built an automated pipeline converting podcasts into SEO-optimized articles and social content; post-trained a 3B open model with QLoRA SFT + DPO on Kaggle T4 GPUs, reducing hallucinated entities by X% and improving SEO compliance from Y% to Z% versus the base model.

Fill in X/Y/Z from `reports/results.md`; only claim numbers you measured.
