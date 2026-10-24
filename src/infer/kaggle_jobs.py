"""Inference jobs on Kaggle (kernel 07_infer, sent by scripts/kaggle_infer.py): podcast links or
transcripts -> articles with the trained adapters. Nothing is trained.

A job is {"id", "url", "episode", "transcript", "title", "keyword", "model"}, model one of MODEL_DIRS, or
has "audio": a file name in the attached pod2post-uploads dataset (a listener's upload) instead of a url.
Two stages, because faster-whisper needs the CUDA 12 torch preinstalled on Kaggle and vLLM installs its own:
  1. `python -m src.infer.kaggle_jobs transcribe jobs.json out/ [uploads/]` downloads and transcribes every
     link or upload job (GPU, configs/asr.yaml) into out/<id>/transcript.json, in a subprocess, before vLLM is installed;
  2. write_all() runs src/infer/article.py's generate() (the website's code) per job with vLLM and writes
     out/<id>/result.json (+ article.md). A failed job records its error and the others still run.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tarfile
import time
from pathlib import Path

MODEL_DIRS = {"sft_dpo": "sft-dpo", "sft": "sft", "base": None}  # job model -> adapter folder (None = base)
SEP = "__"  # dataset files are "<adapter>__<file>": Kaggle datasets skip sub-folders
ENV_PACKAGES = ["vllm", "textstat", "spacy", "en_core_web_sm"]  # what stage 2 needs beyond Kaggle's image
ENV_TAR = "vllm-env.tar"  # kernel 07_deps: ENV_PACKAGES installed with pip --target, as one tar
# vLLM start-up: CUDA graphs for the batch sizes the pipeline uses (Stage A sends all of an episode's chunks
# at once; more than 16 wait their turn), not vLLM's default ~67 sizes. 07_infer-v1 spent 97 s capturing.
ENGINE = {"max_num_seqs": 16, "compilation_config": {"cudagraph_capture_sizes": [1, 2, 4, 8, 16]}}


def use_prebuilt_env(tar: Path | None, dest: Path = Path("/tmp")) -> bool:
    """Put 07_deps' packages first on the path (this process and vLLM's workers); False -> pip instead."""
    if tar is None or not tar.is_file():
        return False
    t0 = time.time()
    with tarfile.open(tar) as f:
        f.extractall(dest, filter="tar")
    site = str(dest / "vllm-env")
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([site, os.environ.get("PYTHONPATH", "")]).rstrip(os.pathsep)}
    ok = subprocess.run([sys.executable, "-c", "import vllm, spacy, textstat; spacy.load('en_core_web_sm')"],
                        env=env).returncode == 0
    if ok:
        os.environ["PYTHONPATH"] = env["PYTHONPATH"]
        sys.path.insert(0, site)
    print(f"[env] pre-built packages {'in use' if ok else 'do not import; installing with pip'} "
          f"({time.time() - t0:.0f} s)", flush=True)
    return ok


def transcribe(jobs: list[dict], out: Path, uploads: Path | None = None) -> None:
    from src.asr.url_audio import resolve
    from src.infer.article import Transcriber

    transcriber = Transcriber()
    for job in jobs:
        d = out / job["id"]
        d.mkdir(parents=True, exist_ok=True)
        if not ((job.get("url") or "").strip() or job.get("audio")):
            continue
        t0 = time.time()
        try:
            if job.get("audio"):
                path = (uploads or Path("/nonexistent")) / job["audio"]
                if not path.is_file():
                    raise FileNotFoundError(f"uploaded file {job['audio']} is not in the attached dataset")
                got = {"path": path, "title": None, "kind": "upload", "audio_url": None}
            else:
                got = resolve(job["url"], d / "download", episode=job.get("episode") or "")
            t1 = time.time()
            utterances, minutes = transcriber(str(got["path"]))
            rec = {"utterances": utterances, "minutes": minutes, "title": got["title"], "kind": got["kind"],
                   "audio_url": got["audio_url"], "download_seconds": round(t1 - t0),
                   "transcribe_seconds": round(time.time() - t1)}
        except Exception as e:  # a bad link fails its own job only
            rec = {"error": f"{type(e).__name__}: {e}"}
        finally:
            shutil.rmtree(d / "download", ignore_errors=True)
        (d / "transcript.json").write_text(json.dumps(rec, ensure_ascii=False), encoding="utf-8")
        print(f"[transcribe] {job['id']}: " + (rec.get("error") or f"{rec['minutes']:.1f} min of audio in "
                                                f"{rec['transcribe_seconds']} s"), flush=True)


def unpack_adapters(src: Path, dest: Path) -> dict[str, str]:
    """Flat dataset files -> one folder per adapter (symlinks), as vLLM expects."""
    paths = {}
    for f in sorted(src.iterdir()):
        name, sep, file = f.name.partition(SEP)
        if sep:
            (dest / name).mkdir(parents=True, exist_ok=True)
            if not (dest / name / file).exists():
                (dest / name / file).symlink_to(f)
            paths[name] = str(dest / name)
    return paths


def write_all(jobs: list[dict], out: Path, adapters: dict[str, str], cfg: dict, gen=None) -> dict:
    """Generate every job's article; returns {id: {"ok", "seconds" | "error"}}."""
    from src.infer.article import generate

    if gen is None:
        from src.infer.generator import VLLMGenerator

        g = cfg["generation"]
        gen = VLLMGenerator(cfg["base_model"], tensor_parallel_size=g["tensor_parallel_size"],
                            max_model_len=g["max_model_len"], gpu_memory_utilization=0.85,  # no NLI scorer here
                            max_lora_rank=16, engine=ENGINE, temperature=g["temperature"], top_p=g["top_p"],
                            top_k=g["top_k"])
    summary = {}
    for job in jobs:
        d = out / job["id"]
        d.mkdir(parents=True, exist_ok=True)
        path = d / "transcript.json"
        rec = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None
        try:
            if rec and "error" in rec:
                raise RuntimeError(rec["error"])
            name = MODEL_DIRS[job.get("model") or "sft_dpo"]
            gen.use_adapter(name, adapters[name] if name else None)
            fetch = transcriber = None
            if rec:  # stage 1 already downloaded (or was given) and transcribed it
                def fetch(url, dest, episode=""):
                    return {"path": "transcribed", "title": rec["title"], "kind": rec["kind"],
                            "audio_url": rec["audio_url"]}

                def transcriber(_path):
                    return rec["utterances"], rec["minutes"]
            res = generate(gen, job.get("title") or "", job.get("transcript") or "", None, job.get("keyword") or "",
                           url=(job.get("url") or "") if not job.get("audio") else "uploaded audio",
                           episode=job.get("episode") or "",
                           transcriber=transcriber, fetch=fetch)
            scores = json.loads(res["scores"])
            if rec:
                scores["source"].update(download_seconds=rec["download_seconds"],
                                        transcribe_seconds=rec["transcribe_seconds"])
            res["scores"] = json.dumps(scores, indent=2)
            if res["file"]:
                shutil.copy(res["file"], d / "article.md")
                res["file"] = "article.md"
            summary[job["id"]] = {"ok": True, "seconds": scores["generation_seconds"]}
        except Exception as e:
            res = {"error": f"{type(e).__name__}: {e}"}
            summary[job["id"]] = {"ok": False, "error": res["error"]}
        (d / "result.json").write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[write] {job['id']}: {summary[job['id']]}", flush=True)
    return summary


if __name__ == "__main__":
    if len(sys.argv) not in (4, 5) or sys.argv[1] != "transcribe":
        raise SystemExit("usage: python -m src.infer.kaggle_jobs transcribe jobs.json out/ [uploads/]")
    transcribe(json.loads(Path(sys.argv[2]).read_text(encoding="utf-8")), Path(sys.argv[3]),
               Path(sys.argv[4]) if len(sys.argv) == 5 else None)
