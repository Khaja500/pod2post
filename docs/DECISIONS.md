# Decisions

Non-obvious choices, newest last (spec §11 rule 6).

## 2026-10-20 · Phase 0

**D1. This folder is its own git repo.** The enclosing git repository was rooted at the home directory, so any commit here would have tracked unrelated files. `git init` here isolates the project. The GitHub remote (`pod2post`, private) is still to be created by the user.

**D2. Kaggle auth follows Kaggle CLI 2.2.4, not the legacy `kaggle.json` flow.** CLI 2.x prefers, in order: an API token (`KAGGLE_API_TOKEN` env var or `~/.kaggle/access_token`), a legacy key (`kaggle.json` or `KAGGLE_USERNAME` + `KAGGLE_KEY`), then OAuth (`kaggle auth login`, cached in `~/.kaggle/credentials.json`). All of them still work. `scripts/kaggle_run.py` shells out to the `kaggle` executable and never imports the `kaggle` package, because importing it authenticates eagerly and calls `exit(1)` when no credentials exist. The script only checks that a credential file or env var *exists* and never opens it.

**D3. The cloud allowlist needs `api.kaggle.com`.** The CLI 2.x SDK sends API calls to `https://api.kaggle.com`, a host plan.md does not list. Output downloads and dataset uploads go through signed URLs, which are presumed to be on `storage.googleapis.com`. `kaggle_run.py check` probes each host.

**D4. Accelerator: `machine_shape: "NvidiaTeslaT4"`, then pin one GPU in code.** Kaggle's current list (Sep 2026) names `NvidiaTeslaT4` as the default GPU, a 2×T4 machine, and no longer lists P100. A single-T4 shape isn't offered, so kernels call `kaggle_env.pin_single_gpu()` (`CUDA_VISIBLE_DEVICES=0`) before importing torch, as spec §2 requires. The T4 is sm_75 with no native bf16, so training uses fp16. *Verified by 00_smoke v1:* nvidia-smi shows 2× Tesla T4, torch sees one, capability (7, 5), `native_bf16: false`, and the kernel is private on `NvidiaTeslaT4`.

**D5. Smoke runs push to a separate kernel slug (`<slug>-smoke`).** Downstream kernels attach the *latest* version of an upstream kernel via `kernel_sources`. If smoke and full runs shared a slug, a smoke run would silently replace the full SFT adapter that `04_dpo` attaches. Smoke builds therefore map every project `kernel_sources` entry to its `-smoke` twin. `00_smoke` is smoke-only and keeps one slug.

**D6. Notebooks are generated into `build/` (git-ignored), not `kaggle/<k>/run.ipynb`.** Each kernel folder holds `kernel-metadata.json`, a template with a `<KAGGLE_USERNAME>` placeholder, and `main.py`, which uses percent-format cells. `build_notebook.py` renders the metadata and bundles `src/`, `configs/` and `requirements-kaggle.txt` into the notebook as one cell. The code therefore needs no dataset upload, and every push carries exactly the committed code. `kernel_sources` and `dataset_sources` are still used for data and upstream outputs.

**D7. The smoke-first rule is enforced by a code hash.** The hash covers `main.py`, the metadata template, the builder, and every bundled file, and does not depend on smoke or full mode. `push --full` is refused unless a smoke run of the same kernel with the same hash completed. Other refusals: the target kernel is queued or running (duplicate session), an upstream kernel is not `complete`, or any other project run is active during a full push. `--force` overrides each check.

**D8. Quota log durations.** Each kernel writes `run_manifest.json`, which records its own start and end times and runtime. Each quota row records both the in-kernel runtime and the push-to-finish wall time. The weekly total uses the runtime where available and otherwise falls back to wall time, which is an upper bound. `kaggle quota`, Kaggle's own counter, is shown by `check` and `quota` and is authoritative.

**D9. `wait` ignores a terminal status in the first 3 minutes after a push unless it has seen the run active.** `kaggle kernels status` ignores the version number and reports the *latest* run. Right after a push, that can still be the previous version's `complete`.

**D10. Pulled outputs go to `runs/<date>/<kernel>[-smoke]-v<version>/`.** The spec's layout is `runs/<date>/<kernel>/`. The version suffix stops two runs on the same day from mixing files.

**D11. Version pins are provisional.** unsloth 2026.10.7 requires `transformers<=5.17.0`, `trl<=1.15.0` and `datasets<5`, so the newest releases (5.20.2, 1.16.0, 5.1.0) would conflict. `requirements-kaggle.txt` pins the newest versions inside those bounds and leaves torch at the Kaggle image's version. Unsloth's docs say to let `pip install unsloth` resolve dependencies, and its Kaggle example notebooks pin much older versions (transformers 4.57.6, trl 0.22.2). `00_smoke` installs the pins and writes `pip_freeze.txt`.

  *Verified 2026-10-20 (00_smoke v1):* every pin installed and imported on Kaggle. The image is Python 3.13.15 with preinstalled torch 2.11.0+cu128, left unchanged. Notable transitive versions: torchao 0.18.0, xformers 0.0.35, triton 3.6.0, tokenizers 0.23.1, huggingface_hub 1.29.0, cut-cross-entropy 25.1.1, ctranslate2 4.8.3. The kernel took 93 s including the install.

**D12. Kernels `01`–`05` are Phase 0 placeholders.** They only prove chaining: each reads every upstream kernel's manifest, and `03`/`04` write a checkpoint so resume can be tested. Each is replaced in its phase. `dataset_sources` stay empty until Phase 1 creates `pod2post-data`.

**Open: resume by attaching a kernel's own output (`push --resume`).** This works in the local simulation. Whether Kaggle accepts a kernel listing *itself* in `kernel_sources` is not yet verified. If it doesn't, the fallback is to pull the checkpoint and re-upload it as a private dataset.

**Base model:** see D16.

## 2026-10-20 · Phase 0 (after first Kaggle run)

**D13. CLI 2.x returns `Permission 'kernels.get' was denied` (not 404) for a kernel that doesn't exist.** Under our own username, `kaggle_run.py` treats that response as "absent".

**D14. `UNSLOTH_COMPILE_LOCATION=/tmp/unsloth_compiled_cache`.** By default Unsloth writes about 30 compiled-trainer files into the current directory, which is `/kaggle/working`, so they would ship with every output. `src/kaggle_env.py` sets the env var on import, and kernels import it before unsloth.

**D15. Teacher option 1 (free open model on Kaggle).** The user chose it on 2026-10-20. Kernel `02_teacher_gen` therefore runs, and no API key is needed anywhere. Generation costs about 6 GPU-hours, which comes out of the 20 h budget.

## 2026-10-20 · Phase 1

**D16. Base model: `Qwen/Qwen3-4B-Instruct-2507`, loaded as `unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit`.** Candidates were checked on the HF API on 2026-10-20:
- **Qwen3.5-4B** (Feb 2026): rejected. It is a hybrid model, with 24 of 32 layers Gated-DeltaNet linear attention, and it is multimodal with 4.66B parameters. Linear-attention kernels on a T4 (sm_75) in fp16 are a real risk.
- **Granite-4.2-3B** (Aug 2026): rejected for now. It is dense, 3.66B and Apache-licensed, but it is a reasoning ("thinking") model with no Unsloth build yet.
- **SmolLM3-3B**: the fallback. It is dense, 3.08B and Apache-licensed.
- **Qwen3-4B-Instruct-2507**: chosen. It is dense (`Qwen3ForCausalLM`), non-thinking, Apache-2.0, ungated, with 256k context and first-class Unsloth T4 support.

It has 4.0B parameters, slightly above the spec's "~3B", so the headline should say "4B". The choice is cheap to revisit before Phase 4, because teacher data doesn't depend on the student. Only chunk sizes use this tokenizer, and they would shift by a few percent. The user delegated the choice by asking for full automation.

**D17. Data choices.**
- **MediaSum source:** the `ccdv/mediasum` validation zip (34 MB, 10k interviews), not the 1.4 GB training split. 2,000 interviews are plenty, and we don't evaluate on MediaSum, so its split names don't matter.
- **Long interviews are kept on purpose** (all 225 over 6k words). Test episodes have a median of 11 chunks, while a uniform MediaSum sample would have a median of 2. Without them, Stage B would never see many-chunk notes.
- **Test set:** 38 TalkRL + 12 Changelog. The TalkRL trailer was swapped for Changelog #662. This was re-frozen minutes after the first freeze, before anything was derived from it.
- **Changelog source:** the CC BY-SA 4.0 originals, not the HF copies.
- **SPoRC is skipped** for now: it is gated and there is no HF login. See SOURCES.md.

**D18. Cleaning.**
- Fillers ("um", "uh", "hmm"…), stutters ("I I"), markdown escapes and links, and stage directions ([laughs]) are removed.
- **No re-casing.** Capitalising each Whisper segment broke mid-sentence splits.
- **Segment merging:** consecutive same-speaker segments are merged into paragraphs that end a sentence (40+ words, capped at 200). Each paragraph keeps its first timestamp, so Stage A can cite times without a timestamp on every line.

**D19. Chunker.** Chunks are at most 1,500 tokens with the base tokenizer, with about 150 tokens of overlap. They break between utterances, falling back to sentence and then word breaks for over-long utterances.

## 2026-10-20 · Phase 2

**D20. Teacher: `Qwen/Qwen3-14B-AWQ` on vLLM 0.32.0, tensor-parallel across both T4s, thinking disabled.**
- It is the official 4-bit AWQ build, Apache-2.0 and ungated.
- vLLM's docs list compute capability 7.5 (T4) as supported.
- Gemma-4-12B was rejected: Gemma models have a history of fp16 overflow, and the T4 has no bf16.
- vLLM pins torch 2.13.0, so it is installed only in the kernels that use it. Every kernel now installs only the packages it needs (`install_requirements(only=...)`).

**D21. ASR: faster-whisper `large-v3-turbo`, fp16, batched (16), VAD on.**
- PyAV is pinned to 18.1.0. Kaggle's PyAV 19 removed the `metadata_errors=` argument that faster-whisper 1.2.1 passes, and the first smoke run failed on it.
- Audio: 5 CC episodes from the Internet Archive (CC BY / BY-SA), plus Changelog #661 for an objective WER check against its human transcript. #661 is not in the test set, and its audio is used privately only.

**D22. Quota accounting check.** After 3 runs, `kaggle quota` showed 0.05 h against 0.032 h in the local log, so each session carries about 20 s of startup overhead the log doesn't capture. Queue time is not counted. `kaggle quota` remains authoritative.

**D23. Stage B is sized to the 4,096-token training length.**
- The constraints: training sequences are capped at 4,096 tokens, and DPO caps the prompt at 2,560 (spec §2, §5.4).
- Merged notes now get a 1,800-token budget, inside the spec's ≤2,500.
- The article is 700–1,000 words and the social pack is tighter.
- Prompt plus response comes to about 3.9k tokens.
- `build_sft` drops any example still over 4,096 and logs the count.
- At 900–1,400 words, a Stage B example came to about 5k tokens, which would have forced either truncation or a longer max length than a T4 can train.

**D24. Kernels that install vLLM uninstall torchaudio.** vllm 0.32.0 installs torch 2.13, a CUDA 13.0 build. Its `torchaudio==2.11.0` pin is satisfied by Kaggle's preinstalled 2.11.0+cu128, so pip leaves that in place. torchaudio then raises a CUDA-version mismatch at import, and `transformers.audio_utils` imports torchaudio whenever it is installed. None of these kernels process audio, so they remove it. This was found by the first teacher smoke run.

## 2026-10-21 · End-to-end pipeline

**D25. `kaggle_run.py pipeline` runs the whole project.** It runs 01 → 05 and does the local steps in between:
- After 02: filter, `build_sft`, `build_dpo`, then dataset upload. In full mode it also enforces the Phase 3 acceptance rule (≥70% of target survives).
- `--resume` is used whenever a run stopped at its time budget.
- WER is computed after 01 (full mode) and the report is written after 05.

Smoke runs read `pod2post-data-smoke`, so files built from smoke data can never reach the real dataset. Smoke runs of the vLLM kernels (02, 04, 05) get 50 minutes instead of 20, because install, model download and engine start take 8–12 minutes before any work.

**D26. Teacher review findings, fixed before the full generation.**
- The teacher expanded names ("Donald Trump" where the transcript says "Trump"). The prompt now requires names copied exactly.
- It spliced two speakers' lines into one quote. Quotes must now be one continuous passage from a single speaker.
- It wrote timestamps as the string "null". These are parsed as empty.
- It overused the keyword: a density of 2.7% against the 2.5% ceiling. The prompt now caps it at 10 uses.
- It closed the social-pack JSON early. A narrow repair applies to teacher data only, and evaluation stays strict.
- Stage B grounding is now checked against the transcript only, matching evaluation, so entities invented in the notes can't reach training data.
- Hashtags such as #BarackObama count as grounded when the name is in the transcript.

**D27. The readability prompt was fixed before the full generation.** The smoke evaluation showed Flesch reading ease of about 20 for every system, and the teacher's own articles scored 30–43 against the spec's 50–70 target. The measurement was checked: prose-only scoring gives the same values. The Stage B prompt now asks for plain, conversational English: 12–16 words per sentence, everyday words, and technical terms explained. Flesch is now scored on prose sentences only, excluding headings and FAQ questions.

**D28. `pipeline --full` enforces the smoke-first rule.** Before each full stage, if no completed smoke run of exactly the current code hash exists, the pipeline runs the smoke version first. Spec rule 4 is therefore automatic, not a manual step.

**D29. Time budgets leave a margin before Kaggle's 3-hour hard limit.** A killed run may lose its outputs. The budgets are: teacher 2 h 15 m (after install and model load, with room for the final batch), SFT 2 h 30 m, and DPO 2 h of training (SFT sampling and the vLLM venv run first). A run that stops at its budget is resumed automatically.

**D30. Robustness learned from the smoke runs.**
- Status checks retry local network outages with backoff (a DNS failure on the laptop once killed a wait).
- Downloads retry connection resets.
- `wait` treats a transient "absent" answer for a kernel we pushed as a retry.
- Every local data copy is verified by hash, after a disk fault zeroed a 16 KB block of one copy.

## 2026-10-22 · Phase 3 result

**D31. Entity grounding now accepts normal name variants.** The first full teacher run kept only 168 of 508 Stage B articles, below the ≥280 acceptance bar. 55% were flagged for an "invented entity", but sampling showed most were false positives:
- country forms, abbreviations and demonyms ("U.S.", "the United States", "American" when the speaker said "America")
- acronyms in either direction ("ACA" ↔ "Affordable Care Act")
- full names when the transcript has only the surname ("Donald Trump" for "Trump")
- glued lowercase hashtags ("#trentonduckett")
- NER noise ("n’t")

`grounded_entity` now accepts aliases, stem variants, acronyms, and full names or phrases whose head word plus at least half their words are in the source. Hashtags are not checked. An entity sharing nothing with the transcript is still ungrounded. The entities still flagged are real additions from world knowledge (Amy Coney Barrett, Sarah Palin, the NFL, dates the transcript never states).

Re-filtering the same output, with no new generation: Stage A kept 1,331 of 1,861 and Stage B 315 of 508. Both are above the acceptance bar. Evaluation uses the same function, so every system is scored by the same rule.

**D32. Kernels that launch `src` modules as subprocesses include them in their code hash.** `04_dpo` runs `-m src.train.sample_sft`, which the import scan missed.

**D33. Phase 2 results.** Six episodes (about 6 h of audio) were transcribed at about 60× real time (real-time factor 0.017). Raw WER against Changelog #661's human transcript is 21.1%. Of that, 16.9 points are insertions: Whisper keeps the fillers and false starts the edited reference removes. Substitutions plus deletions come to 4.1%.

**D34. "Z× cheaper than an API model" is a free estimate, at the user's request.** No API is called. `src/eval/cost.py` computes:
- **Prompt tokens:** rebuilt exactly from the test transcripts and each system's own merged notes.
- **Completion tokens:** as measured on Kaggle.
- **API cost:** tokens × published per-token prices (OpenRouter's public model list, checked 2026-10-21, kept in `configs/costs.yaml`).
- **Self-hosted cost:** measured seconds per episode on one T4 × $0.526/h, the AWS g4dn.xlarge on-demand rate per instances.vantage.sh on 2026-10-21. Kaggle's free tier is shown as $0 but is not used for the ratio.

The one-time training cost is reported separately from the quota log. The caveat: an API model would emit somewhat different token counts. spec.md and plan.md were updated to match the build (2026-10-22).

**D35. 04_dpo samples the SFT model only where a pair can use the result, under a time budget.** The first full run (v1, 2026-10-22) sampled all 315 prompts × 2 and hit the 3 h cap with no output: vLLM serves the base in fp16, so the T4's KV cache holds only ~5 long sequences and throughput is ~30 tok/s. `assemble()` tries an SFT failure first on every 4th prompt only, so `sft_failure_slots()` picks those 75 prompts (same order); `sample_sft` sends them 4 at a time and starts no batch after `sample_time_budget_s` (1 h). DPO training's budget went from 2 h to 1.5 h so install + sampling + training fit 3 h. Prompts the budget leaves unsampled fall back to the next rejected type.

**D36. GPU budget raised from 20 h to 30 h (user, 2026-10-22).** 14.8 h were used by then, 3.0 of them by the failed DPO run, and the full eval alone needs ~8 h.

**D37. 05_eval runs in resumed batches.** Scaled from the smoke eval (2 shortest episodes, 10 chunks; the test set has 630), generation takes ~3.4 h for base and ~4.9 h each for SFT and SFT+DPO (LoRA in vLLM: ~28 vs ~43 tok/s), so no single system fits the 3 h cap and per-system kernels would not help. Episodes now go through the pipeline 4 at a time; each batch is appended to `outputs/<system>.jsonl` with summed stats in `outputs/<system>.stats.json`, no batch starts after 1h50m, and a system is scored as soon as it is complete. `--resume` copies outputs/ and scores/ forward; `run_stage` resumes `05_eval` up to 8 times on `results.json`'s `stopped_by_time_budget`. The last run's results.json and outputs hold every system, so the report and 06_judge are unchanged. Seconds per episode are summed over batches of 4, slightly above an all-at-once run.

**D38. The eval generates on both T4s, and SFT + DPO writes from SFT's notes (user, 2026-10-23).** With 12.5 of the 30 GPU-hours left, the D37 eval (~15 h on one T4) did not fit. Stage A is ~85% of its time (630 chunks vs 50 articles). Two changes:
- **Tensor parallel 2.** Every Kaggle run already has two T4s and Kaggle bills session time (14.6 h counted vs 14.8 h of logged runs, all on 2× T4), so the second GPU is free. Splitting the fp16 weights frees each GPU for KV cache: ~5× the concurrent sequences of one T4. `gpu_memory_utilization` is 0.70 with NLI batches of 8: at 0.80 vLLM overshot by ~1 GB (its CUDA-graph estimate) and the scorer ran out of memory on GPU 1 (smoke v3). Measured in that smoke run: base generation 156.6 s for the 2 smoke episodes vs 294.7 s on one T4 (1.9×; 80 vs 43 tok/s), KV cache 73,696 tokens vs ~21k. Times are reported per episode (wall) and as T4-seconds (wall × 2) for the cost estimate.
- **Notes reuse.** SFT trained on both stages (600 Stage A, 314 Stage B); DPO trained on Stage B prompts only and moved the adapter weights by at most 0.005. SFT + DPO therefore takes SFT's merged notes and keywords and runs Stage B only, which also confines the SFT vs SFT + DPO difference to the article step. Its tokens/s, s / episode and Stage A invalid rate are SFT's or Stage-B-only and are labelled so in the report; the cost estimate uses SFT, the system that ran the whole pipeline.

Estimate: ~5–5.5 GPU-hours for the full eval, in 2–3 resumed runs (8 episodes per batch, 2 h budget). Rejected: 25 of the 50 test episodes (changes the frozen test set and doubles the noise).

Measured (2026-10-23): the full eval took 3.6 GPU-hours in 2 runs (2.28 h + 1.27 h), under the estimate. Per-episode generation wall time on 2 T4s: base 92.6 s, SFT 103.7 s, SFT + DPO 24.1 s (Stage B only).

**D39. DPO is retrained on the SFT model's own samples, scored with the eval's faithfulness scorer.** The full eval (`runs/2026-10-23/05_eval-v2`) missed both training targets:
- **SFT vs base:** SEO 97.8% vs 80.5% and fewer articles with an invented entity or number (26% vs 44%), but faithfulness 38.6% vs 43.5%. Paired over the 50 episodes, the gap is −4.9 points (bootstrap 95% CI −8.5 to −1.3) and SFT is higher on 13 of 50. Flesch 29.7 vs 48.7.
- **SFT + DPO vs SFT:** within noise on every metric (faithfulness −0.4 points, CI −3.5 to +2.9; invented entities 0.48 vs 0.42). 265 of its 285 pairs were synthetic (175 injected hallucinations), which the model separates easily (validation reward accuracy 100%), and the adapter moved at most 0.005 from SFT.

Checks before spending GPU time:
- **Not a sentence-length artefact.** The fine-tuned models write longer sentences (20 vs 12 words), but within each system longer-sentence articles score slightly *higher* (regression: +0.007 per word; the SFT gap grows to −10 points when length is controlled).
- **The teacher data (local check on the Mac's GPU; 3 SFT test articles re-scored there match Kaggle's scores exactly, in fp32 and fp16).** 49 randomly chosen teacher Stage B articles: faithfulness 54.1% (95% CI 50.7–57.5%), Flesch 40.7 (6 of 49 in 50–70), 19.6 words per sentence, SEO 91%, 1 invented entity in total. The filter (§4.3) checks entities, numbers and format, never sentence-level support, and the student inherited the teacher's long sentences. These are MediaSum news interviews; the test set is tech podcasts, so 54.1% and 38.6% are not directly comparable. The DPO run below measures the teacher and the SFT model on the same prompts. Files: `runs/2026-10-23/local_teacher_check/`.

Changes (`04_dpo`, D35's sampling replaced):
- **On-policy pairs scored by the metric itself.** `sample_sft` samples the SFT model twice per prompt on both T4s (tensor parallel, groups of 16 prompts, no new group after 50 min), in the pair order. `score_samples` then scores each sample and the teacher's answer with `score_episode` (NLI faithfulness, invented entities/numbers, SEO, Flesch, strict parsing) plus the training filter's checks, on a free T4. Pair (a): rejected = the least faithful sample; chosen = the most faithful answer passing every check (teacher or the other sample), kept if ≥ 8 points more faithful, no lower on SEO and ≥ 85% as long. Synthetic pairs (b)–(d) fill a third of the total on the other prompts, so the entity-grounding signal stays but no longer dominates.
- **learning_rate 5e-6 → 1e-5**, 1 epoch. Training gets whatever is left of 2h45m after sampling and scoring, at most 1.5 h, so the run cannot reach the 3 h cap.
- **The eval regenerates only what changed.** `05_eval --resume` now drops a system whose adapter's training run started after its outputs were made (`adapter_run` in the stats; older stats are compared with the start of the run being resumed), plus the systems that write from its notes. Base and SFT are carried over, so the re-eval is SFT + DPO's Stage B only (~25 min of generation).

Rejected: retraining SFT on the most faithful half of the teacher articles (~2.5 h of SFT, then a full SFT eval with Stage A, ~1.7 h: over the budget left with the judge); deleting unsupported sentences from teacher articles as "chosen" (always shorter, so DPO would learn length; pruning ~45% of sentences leaves ~470-word articles).

**D40. DPO run 2 is retrained for 3 epochs at 2e-5 on its own scored pairs, not evaluated as is.** `runs/2026-10-23/04_dpo-v3` (D39, 1.5 GPU-hours) sampled 128 of the 300 prompts twice in the 50 min budget (139 tok/s on both T4s) and scored them in 14 min:
- **On MediaSum the SFT model is almost as faithful as the teacher.** On the same 128 prompts, both writing from the teacher's notes: teacher 53.8%, SFT samples 51.2% (paired gap 2.6 points, 95% CI 0.7–4.6). Flesch 40.6 vs 40.5, SEO 90.9% vs 91.1%. The SFT samples invent more (0.07 entities and 0.13 numbers per article vs 0.01 and 0.02). Two samples of one prompt differ by 9.3 points on average, more than the teacher gap. On the test podcasts the same model scores 38.6% with its own notes and Flesch 29.7. So most of SFT's loss against base appears out of domain, or comes from its own notes, not from what it learned on MediaSum.
- **Pairs:** 47 faithfulness pairs (chosen 59.4% vs rejected 41.7%; the teacher is chosen in 22) and 23 synthetic, 70 in all. At 8 pairs per step, one epoch is 9 optimizer steps. Training took 20 min of its 90; validation reward accuracy was 0.75 on 4 pairs.
- **The adapter barely moved:** relative L2 change from SFT 0.33% (max 0.0035), half of run 1's 0.64%, which changed no metric. A re-eval (~1 GPU-hour) would most likely be noise.

Change: `num_train_epochs` 1 → 3 and `learning_rate` 1e-5 → 2e-5 (27 steps, ~6× run 2's update; ~1 h of training). A `--resume` run copies the scored samples, so sampling is skipped. It continues the previous checkpoint only when that run was cut off (it saved no `adapter/`); after a finished run, whose checkpoint used another schedule, it trains afresh from SFT. Then the SFT + DPO re-eval and the judge, ~3.5 GPU-hours of the 5.8 left in all.

Rejected: sampling another 128 prompts in the same kernel (+50 min of sampling, +14 of scoring) for ~45 more pairs, which would put the run near 2h45m and leave no margin for the eval and judge; in-domain (tech podcast) training data, the likely real fix, needs new teacher generation and an SFT retrain (~10 GPU-hours).

Run 3 (`runs/2026-10-23/04_dpo-v4`, 1.0 GPU-hour, trained afresh from SFT on run 2's 70 pairs): 27 steps in 60 min. Training loss 0.62 → 0.02 and reward accuracy 1.0 from epoch 2 on, so it fits the 66 training pairs completely; on the 4 validation pairs, loss 0.29, accuracy 0.75 and margin 1.74 (run 2: 0.45, 0.75, 0.61). The adapter moved 1.41% from SFT (max 0.0089), 4× run 2 and 2× run 1.

Eval of run 3 (`runs/2026-10-23/05_eval-v3`, only SFT + DPO regenerated, 0.6 GPU-hours): within noise of SFT again, so Phase 6 fails. Paired over the 50 episodes, against SFT: faithfulness 39.1% vs 38.6% (+0.4 points, 95% CI −2.9 to +3.9), invented entities 0.42 vs 0.42 (CI ±0.12), invented numbers 0.06 vs 0.04, SEO 96.5% vs 97.8% (−1.25 points, CI −3.3 to +0.5), Flesch 28.4 vs 29.7. Four times the weight change did not move the test metrics: preferences learned on 66 MediaSum pairs did not carry over to the tech podcasts.

**D41. An eval-only ablation: the SFT adapter writes Stage B from base's notes (`sft_base_notes`).** Faithfulness is measured against the transcript, so SFT's 4.9-point loss to base could come from its Stage A notes (dropping or blurring what the guests said) or from its Stage B articles (the teacher's long, dense sentences). `sft_base_notes` takes base's merged notes and keywords and runs only Stage B with the SFT adapter. Against base, only the article writer differs; against SFT, only the notes do. If it is at least as faithful as base and keeps SFT's SEO, the pipeline should use base for Stage A and SFT for Stage B, with no training. If it stays near SFT, the articles are the problem and need different training data. Cost: Stage B on 50 episodes plus scoring, ~0.6 GPU-hours plus a smoke run, and the judge then covers it too. Base, SFT and SFT + DPO outputs are carried over.

Result (`runs/2026-10-23/05_eval-v4`, 0.6 GPU-hours): **the loss is in the article writer.** `sft_base_notes` scores 37.1% faithfulness, 0.44 invented entities, SEO 91.0% and Flesch 25.0. Paired over the 50 episodes:
- Same notes, different writer (`sft_base_notes` − base): faithfulness −6.4 points (95% CI −10.6 to −2.5; lower on 32 of 50), +134 words, Flesch −23.8, SEO +10.5 points.
- Same writer, different notes (`sft_base_notes` − SFT): faithfulness −1.5 points (CI −5.2 to +2.3), so SFT's notes are no worse than base's. SEO is 6.8 points lower on base's notes and keywords, in keyword-in-H2 (80% vs 96%) and keyword density (66% vs 90%); base and SFT pick the same primary keyword on only 5 of 50 episodes.
So the base-notes + SFT-article pipeline is not a fix. What SFT learned from the teacher is a longer, denser article (the teacher's own articles have Flesch ~40 on MediaSum and score 54% there); on the tech podcasts that style puts more sentences beyond what the NLI check finds in the transcript. A fix would change the training targets (shorter, plainer, sentence-level-supported teacher articles, ideally on tech podcasts), which means new teacher generation and an SFT retrain, beyond the 30 h budget.

Judge (`runs/2026-10-23/06_judge-v1`, Qwen3-14B-AWQ, 1.05 GPU-hours, each article against base's in both A/B orders): SFT wins 12%, ties 40%, loses 48%; SFT + DPO 14 / 34 / 52; SFT on base's notes 6 / 48 / 46. Base wins in both orders on 23–26 episodes per system and the system on 3–7. Most ties come from position bias: the judge picks the second article in both orders on 15–22 episodes. The judge agrees with the NLI metric: base's plainer articles are preferred. GPU work stops here at 27.9 of 30 h (Kaggle's counter); the 2.1 h left cannot pay for new teacher targets and an SFT retrain.

**D42. The website runs the finished models without new training: llama.cpp on a Mac, CPU or GPU on Cloud Run.** The 4B model plus adapter needs a GPU for vLLM, and the Mac (M1, 8 GB) has none that vLLM supports. So:
- **Mac:** the base model as a pinned 4-bit GGUF (`unsloth/Qwen3-4B-Instruct-2507-GGUF` Q4_K_M, 2.5 GB, sha256 checked) in llama.cpp's `llama-server` (build b11669, Metal), with the SFT + DPO adapter converted to GGUF and picked per request (`src/infer/generator.py` `LlamaCppGenerator`). Measured on a 48-minute episode: 47 minutes (18 transcribing with Whisper `small` on the CPU, 29 writing 9 chunks), about 13 tokens/s written. 4-bit weights are not what was evaluated (fp16 in vLLM), so the eval numbers describe the Kaggle path, not this one.
- **Cloud Run, GPU:** an image with vLLM and both adapters builds (`deploy/cloudrun/Dockerfile`), but Google refused L4 quota automatically on a new billing account (5 requests, 4 regions, before and after upgrading from the free trial). Not deployed.
- **Cloud Run, CPU** (`Dockerfile.cpu`): the Mac set-up in a container, 8 vCPUs, 16 GiB. 4 llama.cpp threads beat 8 by ~20% (443 s vs 552 s on the same transcript). An 8.5-minute episode took 14.5 minutes, 1.5–2× slower than the Mac, and Cloud Run ends requests at 60 minutes, so episodes were capped at 30. Kept as a fallback image.
- **Cost guards on any public deployment:** one instance at most, scale to zero, a per-day article limit in the app (`--daily-limit`, in memory, so it resets when the instance restarts), and a ₹7,500 (~$88) budget alert on the project. Budgets only alert; they do not stop spending.

**D43. Inference on Kaggle's GPUs with the trained adapters (`07_infer`), and the public site sends every job there.** After the weekly quota reset, the owner asked for Kaggle to be used at inference time only. The adapters are frozen as the private dataset `pod2post-adapters` (the files the final eval scored: `04_dpo-v4` and `03_sft-v2`). `scripts/kaggle_infer.py` builds `07_infer` with the jobs in a generated cell, pushes it, waits and pulls the results. In the kernel, links and uploads are downloaded and transcribed first, in a subprocess on Kaggle's own torch (faster-whisper needs CUDA 12, and vLLM installs another torch), then vLLM on both T4s runs `src/infer/article.py`, the website's code, per job; a failed job records its error and the others continue. Uploaded audio goes to the private dataset `pod2post-uploads`, one version per run with older versions deleted. The upload counts as done when the dataset lists the file, because the CLI's exit code was wrong on an upload that succeeded.

Measured (`runs/2026-10-24/07_infer-v1`, 0.27 GPU-hours for both; D44 then cut the start-up): the 48-minute episode took 81 s to transcribe and 3.3 min to write (Mac: 47 min); an 8.5-minute one 11 s and 2.4 min. The run itself spends ~11 minutes on installs and model load, so a single article takes ~15–20 minutes end to end, and batching episodes into one run amortizes the start-up.

The public Cloud Run site (`Dockerfile.kaggle`) is now a 2-vCPU front end with no model: any visitor's link, upload (≤ 32 MB at first, Cloud Run's HTTP/1 request limit; 500 MB since D45) or transcript becomes a `07_infer` run on the owner's account, one at a time, at most 10 a day. The kaggle CLI signs in from a Kaggle API token in Secret Manager, readable only by the service's runtime account. Costs and risks the owner accepted: visitors use the owner's weekly Kaggle GPU hours (10 articles ≈ 3 GPU-hours a day), their audio passes through the owner's private Kaggle dataset, a queue longer than ~3 jobs runs into Cloud Run's 60-minute request limit, and Kaggle is built for notebooks rather than as a backend for a public site, so its terms may not cover this use. These runs are inference outside the 30 h build budget; each is logged in `runs/quota_log.csv`.

Rejected: transcribing on Cloud Run and sending only text (Whisper on CPU is slow, and the GPU run costs the same start-up either way); keeping a Kaggle notebook open as a server (bills GPU hours while idle, and its public link is temporary).

**D44. `07_infer` starts faster: vLLM pre-installed by a CPU kernel, CUDA graphs for 5 batch sizes.** `07_infer-v1`'s log put the ~11 minutes of start-up into: installing vLLM with pip 4.1 min, the engine's start (weights 0.8 min, torch.compile 1 min, CUDA-graph capture for ~67 batch sizes 1.6 min) 3.4 min, Kaggle's queue and Whisper set-up the rest. Changes:
- **`07_deps`** (CPU only, no GPU quota, 9 min) installs the same pinned packages (`vllm`, `textstat`, `spacy`, `en_core_web_sm`) with `pip --target` on Kaggle's Python 3.13.15 and saves them as one 9.1 GB tar. `07_infer` attaches it, unpacks it to `/tmp`, puts it first on `sys.path` and `PYTHONPATH` (vLLM's workers are separate processes), and falls back to pip if it is missing or does not import. Unpacking takes 2 min, not 4.1. Kaggle's own torchaudio must be uninstalled *before* this: with the tar on `PYTHONPATH` the uninstall did not take effect and transformers imported it (CUDA 12.8 vs 13.0 for vLLM's torch), which failed `07_infer-v3`.
- **vLLM engine arguments** (`kaggle_jobs.ENGINE`): `max_num_seqs=16` and CUDA graphs for batch sizes 1, 2, 4, 8 and 16 only. Stage A sends all of an episode's chunks at once (9 for a 48-minute episode; more than 16 wait their turn). Capture went from 66 + 31 s to 43 + 5 s, and the engine start from 202 s to 152 s. Writing the 48-minute episode took 189 s against 200 s before, so the smaller batch limit cost nothing measurable.

Measured through the public site (`07_infer-v4`, the 47.7-minute Libre Lounge episode re-encoded to 64 kbps mono to fit the 32 MB upload limit): 12.3 minutes from upload to article, 0.18 GPU-hours. Not done: bundling the model weights (40 s of download), caching torch.compile between runs (57 s), and keeping a session warm between articles (it would remove the start-up for a burst of jobs, but bills GPU time while idle).

Also: one public test (`07_infer-v2`) hung for 55 minutes and was cancelled by hand; Kaggle kept no log, so the cause is unknown, and the CLI's run time limit (`-t`) did not stop it. The site crashed once when `kaggle_run.py`'s `die()` (a `SystemExit`, written for a terminal) fired inside a request because the image lacked `kaggle/07_deps`. The image now carries it, and the site turns `SystemExit` into an error message.

**D45. The public site serves HTTP/2, so uploads can be 500 MB; YouTube links and failed downloads stop early.**
- **Upload size.** Cloud Run limits HTTP/1 requests to 32 MB, and Gradio sends an upload as one request, so most full episodes (50–70 MB at 128 kbps) were refused. Cloud Run has no such limit over HTTP/2 end to end (`--use-http2`). uvicorn, Gradio's server, speaks HTTP/1 only, so `--http2` mounts the same Gradio app on FastAPI and serves it with Hypercorn 0.17.3 over cleartext HTTP/2 on `$PORT`. The limit is now `MAX_FILE_MB=500`, and the service has 8 GiB, since Cloud Run's disk is memory: the upload is hard-linked, not copied, into the folder that goes to Kaggle. Measured: the original 51 MB, 47.7-minute Libre Lounge mp3 took 13.1 minutes from upload to article through the public site (`07_infer-v7`: 73 s transcribing, 4.0 min writing). The CPU and GPU images keep uvicorn (`--no-use-http2`).
- **YouTube.** A visitor's YouTube link failed after 9 minutes (`07_infer-v5`): YouTube answers data-centre machines with "Sign in to confirm you're not a bot", so yt-dlp cannot download on Kaggle (or Cloud Run). The site now refuses youtube.com and youtu.be links for the Kaggle models with a message to upload the audio or paste the feed instead. The Mac's local models still accept them.
- **Fast failure.** When every job in a run failed to download, `07_infer` now reports that right after stage 1 instead of unpacking and starting vLLM (~5 min).

Rejected: uploading directly from the browser to a Cloud Storage signed URL (no size limit at all, but it needs custom upload code in the page, and HTTP/2 was enough).

**D46. A clearer website layout.**
- **Layout.** A header with the three steps; the episode inputs on the left (title and keyword folded into an accordion), results on the right. A row of score cards turns the Scores JSON into the numbers people look at first, and the social pack is shown as text ready to copy, with each X post's length. A Clear button resets the page, and the download link appears only once there is an article. The theme (Gradio Soft, indigo, Inter) and CSS go to `launch()` and, for the HTTP/2 server, `mount_gradio_app()`, since Gradio 6 takes them there.
- **Sample transcript.** *Try a sample transcript* fills in a short, made-up interview, so a visitor can try the site without an episode. It is filled in by the browser (a `js=` event), so a quick *Create article* click right after it already sends the sample.
- **Bug found while testing.** Gradio 6 does not render a closed accordion, so its text boxes reach the handler as `None`, not `""`, and `generate()` failed on `keyword.strip()`. `run()` now treats missing text as empty, and a test covers it.

Tested on the Mac (llama.cpp, SFT + DPO) with the sample transcript: a 735-word article, all SEO checks passed, 5 X posts, 2.0 minutes of writing. Also checked that the Kaggle front end starts over HTTP/2 with the new theme.
