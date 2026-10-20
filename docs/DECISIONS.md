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

**Deferred: base model.** Spec §5.1 asks for the current best ungated ~3B instruct model. It is chosen before Phase 3 and confirmed with the user (rule 1).

## 2026-10-20 · Phase 0 (after first Kaggle run)

**D13. CLI 2.x returns `Permission 'kernels.get' was denied` (not 404) for a kernel that doesn't exist.** Under our own username, `kaggle_run.py` treats that response as "absent".

**D14. `UNSLOTH_COMPILE_LOCATION=/tmp/unsloth_compiled_cache`.** By default Unsloth writes about 30 compiled-trainer files into the current directory, which is `/kaggle/working`, so they would ship with every output. `src/kaggle_env.py` sets the env var on import, and kernels import it before unsloth.

**D15. Teacher option 1 (free open model on Kaggle).** The user chose it on 2026-10-20. Kernel `02_teacher_gen` therefore runs, and no API key is needed anywhere. Generation costs about 6 GPU-hours, which comes out of the 20 h budget.
