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

### Step 5 · Phase 4: Baseline eval — ✅ done in `05_eval`
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

### Step 6 · Phase 5: SFT — ❌ trained, misses the faithfulness target
🤖 **Prompt:**
> Do Phase 5. Run SFT in SMOKE mode, show me the loss curve, then run the full SFT on Kaggle with checkpointing. When done, run eval on the SFT adapter and add the SFT row to results.md. Update runs/state.json and commit.

✅ **Done when:** SFT beats baseline on faithfulness and SEO score.
❌ **If it doesn't:** check data quality first (Step 4), then learning rate / epochs. Don't just train longer.

*(Eval 2026-10-23: SEO 97.8% vs base 80.5%, but faithfulness 38.6% vs 43.5% (95% CI of the gap −8.5 to −1.3 points) and Flesch 29.7 vs 48.7. Fewer articles invent an entity or number (26% vs 44%): the teacher filter checked entities, numbers and format, never sentence-level faithfulness. Data check and fix: D39.)*

---

## Part 3 — DPO, report, demo (☁️ or 💻)

### Step 7 · Phase 6: DPO — ❌ target missed after 3 runs
🤖 **Prompt:**
> Do Phase 6: build the DPO preference set (spec §4.4) using the SFT adapter's failures plus synthetic rejected answers, then run DPO from the SFT adapter (SMOKE first). Evaluate and add the SFT+DPO row to results.md.

✅ **Done when:** fewer hallucinated entities than SFT and no drop in SEO score.

*(Trained 2026-10-22: 36/36 steps on 285 pairs, val reward accuracy 100%, margin 3.06. The first full run hit the 3 h cap while sampling the SFT model; the sampling now covers only the prompts that use an SFT failure, under a 1 h budget (D35). 61% of the rejected answers are injected hallucinations, so the high reward accuracy may reflect easy pairs; the eval decides.)*

*(Eval 2026-10-23: within noise of SFT on every metric, so the target is missed. Run 2 trains on the SFT model's own samples scored with the eval's faithfulness scorer: least faithful sample vs. most faithful clean answer, plus a third synthetic pairs (D39). Run 2 found SFT nearly as faithful as the teacher on MediaSum (51.2% vs 53.8%) but made only 70 pairs and 9 steps; the adapter moved 0.33%, so run 3 trains on the same pairs for 3 epochs at 2e-5 (D40). Run 3 moved the adapter 1.41% and is still within noise of SFT: invented entities 0.42 vs 0.42, faithfulness 39.1% vs 38.6% (CI −2.9 to +3.9), SEO 96.5% vs 97.8%.)*

### Step 8 · Phase 7: Final eval + report — ✅ done
🤖 **Prompt:**
> Do Phase 7: full eval of base, SFT, and SFT+DPO (plus the teacher as upper bound if budget allows). Add the LLM-judge win rates, cost/latency per episode, and 3 side-by-side example articles to reports/results.md. Write the headline result sentence from spec §1 with real numbers.

*(The judge is kernel `06_judge`. The cost per article is a **free estimate**, not a paid API call: measured tokens × published API prices, compared with T4-seconds at $0.526 per T4-hour. See spec §6.)*

*(How the eval runs: generation on both T4s with tensor parallel; SFT+DPO writes from SFT's Stage A notes, since DPO trained on Stage B only (D38); 8 episodes per batch, saved after each, no new batch after 2 h, and `pipeline` resumes it automatically (D37). Estimated ~5–5.5 GPU-hours, down from ~15 on one T4 with every system running Stage A. Start with `python3 scripts/kaggle_run.py pipeline --full --from 05_eval --skip 06_judge`; it runs the smoke version first.)*

*(Done 2026-10-23: `reports/results.md` from `05_eval-v4` and `06_judge-v1`. The judge prefers base: SFT wins 12% and loses 48% of episodes, SFT + DPO 14% / 52%. An ablation, SFT writing from base's notes, shows the faithfulness loss is in the article writer, not the notes: −6.4 points vs base on the same notes (D41).)*

### Step 9 · Phase 8: Demo — ✅ built and public (2026-10-24)
🤖 **Prompt:**
> Do Phase 8: Gradio app per spec §10. It should accept a pasted transcript or a short audio file and return the article and social pack, plus the faithfulness and SEO scores. Add run instructions to the README for both Kaggle and local.

🧑 **Test it** on 2–3 CC episodes that aren't in any dataset. Screenshot good outputs for your README/portfolio. *(This becomes possible once the DPO adapter exists. The README covers both the Kaggle and local commands.)*

*(Done 2026-10-24. Tested on real links: on the Mac with llama.cpp (a 48-min episode in 47 min), on Cloud Run's CPUs (8.5 min of audio in 14.5 min), and on Kaggle's GPUs with the trained adapters (`07_infer`: the same 48-min episode in 81 s of transcription + 3.3 min of writing, after ~11 min of start-up; 12.3 min from upload to article after D44's start-up cuts). The public Cloud Run site is a front end that sends every link, upload or transcript to Kaggle (D42–D44).)*

### Step 10 · Stretch (only if quota remains)
- ORPO vs SFT+DPO comparison
- GRPO with a format + faithfulness reward (≤ 300 steps)

---

## GPU quota budget

| Work | Planned GPU-hours | Measured so far |
|---|---|---|
| Smoke tests, transcription | ~1.5 | 1.4 (smoke runs of 00–03: 1.3; full transcription: 0.1; later smoke runs are in the rows below) |
| Teacher gen (option 1), baseline, SFT | ~4–10 | teacher 6.1 (full, incl. 2 automatic resumes); SFT 2.8 (full, 1 resume) |
| DPO, final eval, judge | ~4 | DPO 8.3 full (v1: 3.0, no output, hit the 3 h cap while sampling, D35; v2: 2.7; v3: 1.5, D39; v4: 1.0, D40) + 2.1 smoke; eval 4.7 full (3.6 first eval, D38; 1.2 re-evals, D40–D41) + 1.7 smoke; judge 1.05 + 0.2 smoke |
| **Total** | **~10–16** | **27.9** (Kaggle's counter; the local log says 28.3). Building the models is finished |
| Website inference (`07_infer`, from 2026-10-24) | outside the build budget | 0.27 for the first 2 episodes; 0.18 per uploaded 48-min episode after D44; `07_deps` 0.1 CPU-hours (no GPU). Each run is logged in `runs/quota_log.csv` |

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
