#!/usr/bin/env python3
"""Build a Kaggle notebook from version-controlled .py files (spec §7.2).

  python scripts/build_notebook.py 03_sft [--full] [--resume] [--username U]

Inputs:   kaggle/<kernel>/kernel-metadata.json  (template, <KAGGLE_USERNAME> placeholder)
          kaggle/<kernel>/main.py               (percent-format cells: "# %%", "# %% [markdown]")
          src/**/*.py, configs/*.yaml, requirements-kaggle.txt (bundled into the notebook)
Output:   build/kaggle/<slug>/{run.ipynb, kernel-metadata.json}  (git-ignored; never edit by hand)

Smoke builds push to a separate "<slug>-smoke" kernel so a smoke run never
replaces the full-run output that downstream kernels attach.
Stdlib only, so it runs anywhere (local machine or a Claude Code cloud session).
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
KAGGLE_DIR = REPO / "kaggle"
BUILD_DIR = REPO / "build" / "kaggle"
PLACEHOLDER = "<KAGGLE_USERNAME>"
PROJECT_PREFIX = "pod2post-"
SMOKE_DATASETS = {"pod2post-data"}  # smoke runs read pod2post-data-smoke, so smoke-built files never mix in
CELL_RE = re.compile(r"^# %%(.*)$")


def bundle_files() -> list[Path]:
    """Repo files copied into every notebook, sorted for a stable hash."""
    files = [p for p in (REPO / "src").rglob("*.py") if "__pycache__" not in p.parts]
    files += sorted((REPO / "configs").glob("*.yaml"))
    files.append(REPO / "requirements-kaggle.txt")
    return sorted(files)


def kernel_deps(kernel: str) -> list[Path]:
    """src modules main.py imports (transitively) + the configs those files name."""
    main = KAGGLE_DIR / kernel / "main.py"
    seen: set[Path] = set()
    stack = [main]
    # Modules a kernel launches as subprocesses ("-m", "src.train.sample_sft") count too.
    for name in re.findall(r"[\"'](src(?:\.\w+)+)[\"']", main.read_text(encoding="utf-8")):
        cand = REPO.joinpath(*name.split(".")).with_suffix(".py")
        if cand.is_file() and cand not in seen:
            seen.add(cand)
            stack.append(cand)
    while stack:
        for node in ast.walk(ast.parse(stack.pop().read_text(encoding="utf-8"))):
            if isinstance(node, ast.ImportFrom) and node.module and node.module.split(".")[0] == "src":
                names = [node.module] + [f"{node.module}.{a.name}" for a in node.names]
            elif isinstance(node, ast.Import):
                names = [a.name for a in node.names if a.name.split(".")[0] == "src"]
            else:
                continue
            for name in names:
                base = REPO.joinpath(*name.split("."))
                for cand in (base.with_suffix(".py"), base / "__init__.py"):
                    if cand.is_file() and cand not in seen:
                        seen.add(cand)
                        stack.append(cand)
    text = main.read_text(encoding="utf-8") + "".join(p.read_text(encoding="utf-8") for p in seen)
    configs = {p for p in (REPO / "configs").glob("*.yaml") if p.name in text}
    return sorted(seen | configs)


def code_hash(kernel: str) -> str:
    """Hash of everything that affects this kernel's run, independent of smoke/full mode.

    Only the code the kernel actually uses counts, so editing an unrelated module
    does not invalidate another kernel's smoke run.
    """
    h = hashlib.sha256()
    paths = [KAGGLE_DIR / kernel / "main.py", KAGGLE_DIR / kernel / "kernel-metadata.json",
             Path(__file__).resolve(), REPO / "requirements-kaggle.txt", *kernel_deps(kernel)]
    for p in paths:
        h.update(p.relative_to(REPO).as_posix().encode())
        h.update(p.read_bytes())
    return h.hexdigest()[:12]


def load_template(kernel: str) -> dict:
    path = KAGGLE_DIR / kernel / "kernel-metadata.json"
    if not path.is_file():
        raise SystemExit(f"no kernel template: {path.relative_to(REPO)}")
    return json.loads(path.read_text(encoding="utf-8"))


def template_slug(kernel: str) -> str:
    return load_template(kernel)["id"].split("/", 1)[1]


def mode_slug(slug: str, smoke: bool) -> str:
    # 00_smoke is already a smoke-only kernel, so it keeps one slug.
    return f"{slug}-smoke" if smoke and not slug.endswith("-smoke") else slug


def render_metadata(template: dict, username: str, smoke: bool, resume: bool) -> dict:
    meta = json.loads(json.dumps(template).replace(PLACEHOLDER, username))
    slug = mode_slug(meta["id"].split("/", 1)[1], smoke)
    meta["id"] = f"{username}/{slug}"
    meta["title"] = slug  # Kaggle requires the title to slugify to the id
    meta["code_file"] = "run.ipynb"
    sources = []
    for ref in meta.get("kernel_sources", []):
        owner, name = ref.split("/", 1)
        sources.append(f"{owner}/{mode_slug(name, smoke)}" if name.startswith(PROJECT_PREFIX) else ref)
    if resume:
        # Attach this kernel's own previous output so training can resume from its checkpoints.
        sources.append(meta["id"])
    meta["kernel_sources"] = sources
    datasets = []
    for ref in meta.get("dataset_sources", []):
        owner, name = ref.split("/", 1)
        datasets.append(f"{owner}/{name}-smoke" if smoke and name in SMOKE_DATASETS else ref)
    meta["dataset_sources"] = datasets
    return meta


def parse_cells(source: str) -> list[tuple[str, str]]:
    """Split percent-format source into (cell_type, text) pairs."""
    cells, kind, lines = [], "code", []

    def flush():
        text = "\n".join(lines).strip("\n")
        if text.strip():
            cells.append((kind, text))

    for line in source.splitlines():
        m = CELL_RE.match(line)
        if m:
            flush()
            kind, lines = ("markdown" if "[markdown]" in m.group(1) else "code"), []
            continue
        if kind == "markdown":
            line = line[2:] if line.startswith("# ") else line.lstrip("#")
        lines.append(line)
    flush()
    return cells


def config_cell(kernel: str, meta: dict, smoke: bool, resume: bool, digest: str) -> str:
    own = meta["id"]
    upstream = [ref.split("/", 1)[1] for ref in meta["kernel_sources"] if ref != own]
    data = next((ref.split("/", 1)[1] for ref in meta.get("dataset_sources", [])
                 if ref.split("/", 1)[1].startswith("pod2post-data")), None)
    return "\n".join([
        "# Config cell (generated). SMOKE=1: 10 examples, 5 steps, < 10 minutes (spec §7.5).",
        f"KERNEL = {kernel!r}",
        f"SLUG = {own.split('/', 1)[1]!r}",
        f"SMOKE = {int(smoke)}",
        f"RESUME = {resume}",
        f"CODE_HASH = {digest!r}",
        f"UPSTREAM = {upstream!r}  # attached kernel outputs, excluding this kernel",
        f"DATA = {data!r}  # the pod2post-data dataset this run reads (smoke runs use their own copy)",
        "CFG = dict(KERNEL=KERNEL, SLUG=SLUG, SMOKE=SMOKE, RESUME=RESUME, CODE_HASH=CODE_HASH, UPSTREAM=UPSTREAM)",
    ])


def bundle_cell() -> str:
    files = {p.relative_to(REPO).as_posix(): p.read_text(encoding="utf-8") for p in bundle_files()}
    body = "\n".join(f"    {rel!r}: {text!r}," for rel, text in files.items())
    return "\n".join([
        "# Repo code bundled by scripts/build_notebook.py; all logic lives in src/.",
        "import os, sys",
        "from pathlib import Path",
        'BUNDLE_DIR = Path(os.environ.get("POD2POST_BUNDLE_DIR", "/tmp/pod2post"))',
        "_FILES = {",
        body,
        "}",
        "for _rel, _text in _FILES.items():",
        "    _path = BUNDLE_DIR / _rel",
        "    _path.parent.mkdir(parents=True, exist_ok=True)",
        "    _path.write_text(_text, encoding='utf-8')",
        "sys.path.insert(0, str(BUNDLE_DIR))",
        'print(f"bundled {len(_FILES)} files -> {BUNDLE_DIR}")',
    ])


def to_notebook(cells: list[tuple[str, str]]) -> dict:
    nb_cells = []
    for i, (kind, text) in enumerate(cells):
        cell = {"cell_type": kind, "id": f"cell-{i:02d}", "metadata": {},
                "source": text.splitlines(keepends=True)}
        if kind == "code":
            cell.update(execution_count=None, outputs=[])
        nb_cells.append(cell)
    return {
        "cells": nb_cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def build(kernel: str, username: str, smoke: bool = True, resume: bool = False,
          out_root: Path = BUILD_DIR) -> tuple[Path, dict, str]:
    """Write run.ipynb + kernel-metadata.json; return (folder, metadata, code_hash)."""
    meta = render_metadata(load_template(kernel), username, smoke, resume)
    digest = code_hash(kernel)
    main = (KAGGLE_DIR / kernel / "main.py").read_text(encoding="utf-8")
    header = (f"Generated by `scripts/build_notebook.py` from `kaggle/{kernel}/main.py` "
              f"(code hash `{digest}`). Do not edit on Kaggle; edit the repo and push again.")
    cells = [("markdown", header),
             ("code", config_cell(kernel, meta, smoke, resume, digest)),
             ("code", bundle_cell()),
             *parse_cells(main)]
    out = out_root / meta["id"].split("/", 1)[1]
    out.mkdir(parents=True, exist_ok=True)
    (out / "run.ipynb").write_text(json.dumps(to_notebook(cells), indent=1), encoding="utf-8")
    (out / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return out, meta, digest


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("kernel", help="folder under kaggle/, e.g. 03_sft")
    ap.add_argument("--full", action="store_true", help="full run (default: SMOKE=1)")
    ap.add_argument("--resume", action="store_true", help="attach own previous output for checkpoints")
    ap.add_argument("--username", default=os.environ.get("KAGGLE_USERNAME", PLACEHOLDER))
    args = ap.parse_args()
    out, meta, digest = build(args.kernel, args.username, smoke=not args.full, resume=args.resume)
    print(f"built {meta['id']} (code hash {digest}) -> {out.relative_to(REPO)}")


if __name__ == "__main__":
    main()
