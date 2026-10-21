"""Build every kernel and run the generated notebooks locally against a fake /kaggle tree."""
import json
import os
import shutil
import subprocess
import sys

import pytest

import build_notebook as nb

KERNELS = ["00_smoke", "01_transcribe", "02_teacher_gen", "03_sft", "04_dpo", "05_eval"]
# Kernels that can run on a laptop: the environment check and the Phase 0 placeholders.
# Real kernels need a GPU and Kaggle inputs, so they are only syntax-checked here.
LOCAL = [k for k in KERNELS if k == "00_smoke" or "placeholder_stage" in (nb.KAGGLE_DIR / k / "main.py").read_text()]


def code_of(notebook_path):
    nbjson = json.loads(notebook_path.read_text())
    assert nbjson["nbformat"] == 4
    return "\n\n".join("".join(c["source"]) for c in nbjson["cells"] if c["cell_type"] == "code")


def run_notebook(tmp_path, out_dir, meta, upstream_outputs):
    """Execute a built notebook with each kernel_source mounted at /kaggle/input/<slug>."""
    slug = meta["id"].split("/", 1)[1]
    root = tmp_path / f"root-{slug}-{len(list(tmp_path.iterdir()))}"
    (root / "working").mkdir(parents=True)
    for ref in meta["kernel_sources"]:
        src_slug = ref.split("/", 1)[1]
        if src_slug in upstream_outputs:
            shutil.copytree(upstream_outputs[src_slug], root / "input" / src_slug)
        else:  # a real kernel that can't run locally: stand in a completed manifest
            (root / "input" / src_slug).mkdir(parents=True)
            (root / "input" / src_slug / "run_manifest.json").write_text(
                json.dumps({"status": "complete", "code_hash": "stand-in"}))
    script = root / "run.py"
    script.write_text(code_of(out_dir / "run.ipynb"))
    env = {**os.environ, "POD2POST_KAGGLE_ROOT": str(root), "POD2POST_BUNDLE_DIR": str(root / "bundle")}
    proc = subprocess.run([sys.executable, str(script)], cwd=root, env=env, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    manifest = json.loads((root / "working" / "run_manifest.json").read_text())
    assert manifest["status"] == "complete"
    return root / "working", manifest


def test_parse_cells_percent_format():
    cells = nb.parse_cells("# %% [markdown]\n# # Title\n# text\n\n# %%\nx = 1\n\n# %%\n\n# %%\ny = 2\n")
    assert cells == [("markdown", "# Title\ntext"), ("code", "x = 1"), ("code", "y = 2")]


def test_render_metadata_smoke_slugs_and_resume():
    template = nb.load_template("05_eval")
    smoke = nb.render_metadata(template, "tester", smoke=True, resume=False)
    assert smoke["id"] == "tester/pod2post-05-eval-smoke" and smoke["title"] == "pod2post-05-eval-smoke"
    assert smoke["kernel_sources"] == ["tester/pod2post-03-sft-smoke", "tester/pod2post-04-dpo-smoke"]
    full = nb.render_metadata(template, "tester", smoke=False, resume=True)
    assert full["id"] == "tester/pod2post-05-eval"
    assert full["kernel_sources"][-1] == "tester/pod2post-05-eval"
    assert "<KAGGLE_USERNAME>" not in json.dumps(full)
    assert nb.render_metadata(nb.load_template("00_smoke"), "tester", True, False)["id"] == "tester/pod2post-00-smoke"


@pytest.mark.parametrize("kernel", KERNELS)
def test_kernel_templates_follow_spec(kernel):
    meta = nb.load_template(kernel)
    assert meta["is_private"] is True and meta["enable_gpu"] is True and meta["enable_internet"] is True
    assert meta["machine_shape"] == "NvidiaTeslaT4"
    assert meta["id"] == f"<KAGGLE_USERNAME>/{meta['title']}"


def test_code_hash_is_mode_independent(tmp_path):
    _, _, h_smoke = nb.build("03_sft", "tester", smoke=True, out_root=tmp_path)
    out, _, h_full = nb.build("03_sft", "tester", smoke=False, out_root=tmp_path)
    assert h_smoke == h_full
    assert "SMOKE = 0" in code_of(out / "run.ipynb")


@pytest.mark.parametrize("kernel", KERNELS)
def test_every_notebook_compiles(kernel, tmp_path):
    out, _, _ = nb.build(kernel, "tester", smoke=True, out_root=tmp_path)
    compile(code_of(out / "run.ipynb"), f"{kernel}/run.ipynb", "exec")


def test_pipeline_chain_and_resume_run_locally(tmp_path):
    builds = tmp_path / "build"
    outputs = {}
    for kernel in LOCAL:
        out, meta, digest = nb.build(kernel, "tester", smoke=True, out_root=builds)
        working, manifest = run_notebook(tmp_path, out, meta, outputs)
        assert manifest["code_hash"] == digest and manifest["smoke"] is True
        outputs[meta["id"].split("/", 1)[1]] = working

    assert (outputs["pod2post-00-smoke"] / "hello.txt").is_file()
    if "05_eval" in LOCAL:
        assert set(json.loads((outputs["pod2post-05-eval-smoke"] / "run_manifest.json").read_text())["upstream"]) == {
            "pod2post-03-sft-smoke", "pod2post-04-dpo-smoke"}
    if "03_sft" not in LOCAL:
        return
    assert (outputs["pod2post-03-sft-smoke"] / "checkpoints" / "checkpoint-5").is_dir()

    # A resumed run attaches its own previous output and continues from the latest checkpoint.
    out, meta, _ = nb.build("03_sft", "tester", smoke=True, resume=True, out_root=builds)
    working, manifest = run_notebook(tmp_path, out, meta, outputs)
    assert manifest["resume_from"].endswith("checkpoints/checkpoint-5")
    assert (working / "checkpoints" / "checkpoint-10").is_dir()


def test_code_hash_tracks_only_used_code():
    deps = {p.relative_to(nb.REPO).as_posix() for p in nb.kernel_deps("01_transcribe")}
    assert {"src/kaggle_env.py", "src/asr/transcribe.py", "configs/asr.yaml"} <= deps
    assert "src/data/teacher.py" not in deps and "configs/teacher.yaml" not in deps
    teacher_deps = {p.relative_to(nb.REPO).as_posix() for p in nb.kernel_deps("02_teacher_gen")}
    assert {"src/data/teacher.py", "src/data/prompts.py", "src/infer/generator.py", "configs/teacher.yaml"} <= teacher_deps


def test_smoke_builds_read_the_smoke_dataset(tmp_path):
    out, meta, _ = nb.build("03_sft", "tester", smoke=True, out_root=tmp_path)
    assert meta["dataset_sources"] == ["tester/pod2post-data-smoke"]
    assert "DATA = 'pod2post-data-smoke'" in code_of(out / "run.ipynb")
    _, full, _ = nb.build("03_sft", "tester", smoke=False, out_root=tmp_path)
    assert full["dataset_sources"] == ["tester/pod2post-data"]
