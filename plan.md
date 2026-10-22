# plan.md — Pod2Post Execution Plan

`spec.md` says **what** to build; this file says the **order**, **where** (local vs cloud), and **what to type into Claude Code** at each step.

Legend: 🧑 = you do it · 🤖 = Claude Code does it · 💻 = run Claude Code locally · ☁️ = run Claude Code on the web (claude.ai/code)

Total Kaggle GPU budget: ≤ 30 GPU-hours (raised from 20 on 2026-10-22, D36).

> **Revised 2026-10-22.** The steps below run automatically now, with one command per mode (`pipeline` / `pipeline --full`). The per-phase prompts are kept for reference, and each step is marked with its status. Decisions are logged in `docs/DECISIONS.md`.

---

## Part 0 — One-time setup (all 🧑)

- [x] **Kaggle account:** sign up, then **verify your phone number** (required for GPU + internet in notebooks).
- [x] **Kaggle credentials (CLI 2.x):** run `kaggle auth login`, which is OAuth with nothing to copy. Alternatively, save a token from kaggle.com → Settings → API → **Generate New Token** to `~/.kaggle/access_token` and `chmod 600` it. The legacy `kaggle.json` still works. *(Done with `kaggle auth login`; username `khajaaflal`.)*
- [x] **Kaggle CLI:** `pip install kaggle` (2.2.4), then `python3 scripts/kaggle_run.py check`.
- [ ] **GitHub repo:** create a private repo `pod2post` and push. The project is already its own git repo, separate from the home folder, with commits made along the way. Remaining: `gh repo create pod2post --private --source . --push`.
- [x] **Claude Code locally:** opened in the project folder.
- [x] **Teacher option** (spec §4.2): **Option 1**, a free open model on Kaggle (Qwen3-14B-AWQ). *(Chosen 2026-10-20.)*
- [x] **Local Python env:** `python3 -m venv .venv && .venv/bin/pip install -r requirements.txt`, then `.venv/bin/python -m spacy download en_core_web_sm`.

---

## The one-command version (what actually runs now)

```
python3 scripts/kaggle_run.py pipeline          # every stage in SMOKE mode (tiny data)
python3 scripts/kaggle_run.py pipeline --full   # the real run: 01 → 06 + local steps + report
python3 scripts/kaggle_run.py pipeline --full --from 03_sft   # resume from a stage
```

`pipeline --full` handles the following automatically:
- A smoke run of any stage whose code changed since its last smoke run.
- The local steps after 02: filter → build SFT/DPO data → upload, plus the Phase 3 acceptance check.
- Resuming a run that stopped at its time budget.
- The WER check after 01, and `reports/results.md` after 05/06.
- Riding out network blips.

The pipeline stops on any failure it can't fix itself (an acceptance check, an error). Fix the cause and continue with `--from`.

---

## Part 1 — Foundations (💻 local)

### Step 1 · Phase 0: Repo scaffold + Kaggle automation — ✅ done
🤖 **Prompt:**
> Read spec.md and plan.md fully. Do Phase 0 only: create the repo structure from spec §8, the .gitignore (secrets first), requirements files, `scripts/build_notebook.py`, and `scripts/kaggle_run.py` with check/push/status/wait/pull/pipeline commands and the pre-flight guard and quota log. Create a minimal `kaggle/00_smoke` kernel that prints GPU info and writes a file to /kaggle/working. Run `kaggle_run.py check`, then push, wait, and pull the smoke kernel. Stop and show me the result.

✅ **Done when:** the smoke kernel runs on a T4, its output is pulled into `runs/`, and `quota_log.csv` has one row.
🧑 **Check:** open the kernel on kaggle.com and confirm it says "GPU T4" and is private. *(Verified through the API: private, `NvidiaTeslaT4`, capability 7.5, no native bf16. The smoke pipeline later ran green 01 → 05.)*

### Step 2 · Phase 1: Data — ✅ done
🤖 **Prompt:**
> Do Phase 1 from spec.md. First check current licenses for MediaSum, SPoRC, TalkRL, and the Changelog transcript datasets and record them in data/SOURCES.md. Download samples (MediaSum: 2,000 interviews; SPoRC: 500 episodes; all of TalkRL). Freeze the 50-episode held-out test set and save its hashed IDs. Implement and unit-test the chunker (~1,500-token chunks with overlap). Upload processed data as a private Kaggle Dataset `pod2post-data`. Stop and summarize dataset sizes.

✅ **Done when:** `tests/test_chunk.py` passes, the test set is frozen, and the Kaggle Dataset exists.
🧑 **Check:** read 2–3 transcripts yourself. Are they readable? Is the test set made of real podcast episodes? *(Claude read them. That review caused the TalkRL trailer to be dropped, ASR segments to be merged into sentences, and markdown escapes to be removed. SPoRC is gated and was skipped.)*

### Step 3 · Phase 2: Transcription — ✅ done
🧑 ~~Download 5 Creative-Commons podcast episodes~~ → 🤖 Claude picked 5 CC BY/BY-SA episodes from the Internet Archive, plus Changelog #661 as the WER reference.
🤖 **Prompt:**
> Do Phase 2: faster-whisper transcription kernel on Kaggle. SMOKE run first on 2 minutes of audio, then the 5 episodes. Report real-time factor and GPU minutes used.

✅ **Done when:** the 5 transcripts are saved and one has been spot-checked against the audio. *(Done: about 60× real time, 0.2 GPU-h. WER is 21.1% raw against the edited human transcript, but substitutions plus deletions are only 4.1%.)*

---

## Part 2 — Data generation + training (💻 then ☁️)

### Step 4 · Phase 3: Teacher data generation — ✅ done
🤖 **Prompt (option 1, Kaggle teacher):**
> Do Phase 3 with teacher option 1. Write the Stage A and Stage B prompts, generate in SMOKE mode (10 examples) and show me 3 examples of each before the full run. After I approve, run the full generation, then filtering (spec §4.3). Report the drop rate per rule.

🧑 **Important review:** read ~10 generated articles. If they're bad, the trained model will be bad. Fix the prompts now, not later. *(Claude reviewed the smoke outputs and fixed these before the full run:)*
- expanded names
- quotes spliced across speakers
- "null" timestamps
- keyword overuse
- broken social JSON
- hard-to-read prose (Flesch 30–43): the prompt now asks for plain English
- the article was resized so a training example fits 4,096 tokens

✅ **Done when:** ≥ 420 Stage A + ≥ 280 Stage B examples survive filtering. *(The first filter pass kept only 168 Stage B. Most drops were false positives such as "U.S." vs "America", acronyms and surname-only mentions. With the fixed grounding check: **1,331 A / 315 B**. 7.1 GPU-h including smoke runs.)*

### Step 5 · Phase 4: Baseline eval — ⏳ part of `05_eval`
🤖 **Prompt:**
> Do Phase 4: implement the eval suite (spec §6) and its unit tests, then run the base model zero-shot on the test set on Kaggle. Write the baseline row in reports/results.md.

✅ **Done when:** baseline numbers exist for every metric. *(The suite is implemented and tested. `05_eval` scores base, SFT and SFT+DPO in one run, which saves a model load and a queue wait.)*

### ☁️ Switch to the cloud here (optional)
🧑 One-time cloud setup at claude.ai/code:
- [ ] Connect GitHub and select the `pod2post` repo (create it first; Part 0).
- [ ] Create an environment:
  - network access = custom, allowing **`api.kaggle.com`**, `www.kaggle.com`, `kaggle.com` and `storage.googleapis.com` (plus the default package registries)
  - environment variable `KAGGLE_API_TOKEN=...` (or `KAGGLE_USERNAME` + `KAGGLE_KEY`)
- [ ] First cloud prompt: *"Run `scripts/kaggle_run.py check` and tell me if Kaggle is reachable."*

Note: environment variables are visible to anyone who can edit that environment. That's fine for a personal account; never share the environment.

### Step 6 · Phase 5: SFT — ⏳ running
🤖 **Prompt:**
> Do Phase 5. Run SFT in SMOKE mode, show me the loss curve, then run the full SFT on Kaggle with checkpointing. When done, run eval on the SFT adapter and add the SFT row to results.md. Update runs/state.json and commit.

✅ **Done when:** SFT beats baseline on faithfulness and SEO score.
❌ **If it doesn't:** check data quality first (Step 4), then learning rate / epochs. Don't just train longer.

---

## Part 3 — DPO, report, demo (☁️ or 💻)

### Step 7 · Phase 6: DPO — ⏳
🤖 **Prompt:**
> Do Phase 6: build the DPO preference set (spec §4.4) using the SFT adapter's failures plus synthetic rejected answers, then run DPO from the SFT adapter (SMOKE first). Evaluate and add the SFT+DPO row to results.md.

✅ **Done when:** fewer hallucinated entities than SFT and no drop in SEO score.

### Step 8 · Phase 7: Final eval + report — ⏳
🤖 **Prompt:**
> Do Phase 7: full eval of base, SFT, and SFT+DPO (plus the teacher as upper bound if budget allows). Add the LLM-judge win rates, cost/latency per episode, and 3 side-by-side example articles to reports/results.md. Write the headline result sentence from spec §1 with real numbers.

*(The judge is kernel `06_judge`. The cost per article is a **free estimate**, not a paid API call: measured tokens × published API prices, compared with one T4 at $0.526/h. See spec §6.)*

### Step 9 · Phase 8: Demo — ✅ built
🤖 **Prompt:**
> Do Phase 8: Gradio app per spec §10. It should accept a pasted transcript or a short audio file and return the article and social pack, plus the faithfulness and SEO scores. Add run instructions to the README for both Kaggle and local.

🧑 **Test it** on 2–3 CC episodes that aren't in any dataset. Screenshot good outputs for your README/portfolio. *(This becomes possible once the DPO adapter exists. The README covers both the Kaggle and local commands.)*

### Step 10 · Stretch (only if quota remains)
- ORPO vs SFT+DPO comparison
- GRPO with a format + faithfulness reward (≤ 300 steps)

---

## GPU quota budget

| Work | Planned GPU-hours | Measured so far |
|---|---|---|
| Smoke tests, transcription | ~1.5 | 2.9 (all smoke runs: 2.8; full transcription: 0.1) |
| Teacher gen (option 1), baseline, SFT | ~4–10 | teacher 6.1 (full, incl. 2 automatic resumes); SFT 2.8 (full, 1 resume) |
| DPO, final eval, judge | ~4 | DPO full v1: 3.0, no output (hit the 3 h cap while sampling, D35) |
| **Total** | **~10–16** | **14.8 so far**; ~11 more estimated (DPO ~2, full eval ~8, judge) |

Check `python3 scripts/kaggle_run.py quota` (or the Kaggle settings page) before starting a big run.

---

## Rules of thumb while working with Claude Code

1. **One phase per prompt.** Ask it to stop after each phase so you can review. *(Superseded at the user's request: the phases now run back to back. Failures still stop the pipeline.)*
2. **Always SMOKE first.** A broken full run wastes hours of quota. *(Enforced automatically by `pipeline --full`.)*
3. **Look at outputs, not just metrics.** Read generated articles at every phase.
4. **Never paste your Kaggle key into the chat.** It belongs in `~/.kaggle/` (local) or environment variables (cloud).
5. **If a session ends mid-run,** start a new one with: *"Read runs/state.json and plan.md, tell me where we are, and continue from the last completed step."* The pipeline can be continued with `pipeline --full --from <stage>`.

---

## What to put on your resume when done
> Built an automated pipeline converting podcasts into SEO-optimized articles and social content; post-trained a 4B open model with QLoRA SFT + DPO on Kaggle T4 GPUs, reducing hallucinated entities by X% and improving SEO compliance from Y% to Z% versus the base model, at ~N× lower estimated cost per article than a large API model.

Fill in X/Y/Z/N from `reports/results.md`; only claim numbers you measured.
