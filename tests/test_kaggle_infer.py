"""Inference-only Kaggle runs (07_infer): the notebook carries the jobs, adapters unpack into folders,
and each job's result is the website's output (a failed job does not stop the others)."""
import ast
import json
import sys
from pathlib import Path

from src.infer import kaggle_jobs
from tests.test_eval import FakeGen

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import kaggle_infer  # noqa: E402

TRANSCRIPT = "Ann: Zipline delivers blood to hospitals in Rwanda.\nBob: How far do the drones fly?\nAnn: About eighty kilometres."


class SwitchingGen(FakeGen):
    def __init__(self):
        super().__init__()
        self.adapters = []

    def use_adapter(self, name, path=None):
        self.adapters.append((name, path))


def test_notebook_carries_the_jobs_after_the_config_cell(tmp_path):
    jobs = [{"id": "j1", "url": "https://feed", "transcript": "", "episode": "it's \"quoted\"", "title": "",
             "keyword": "", "model": "sft"}]
    out, digest = kaggle_infer.build_with_jobs(jobs, "someone", out_root=tmp_path)
    cells = json.loads((out / "run.ipynb").read_text())["cells"]
    assert "KERNEL = '07_infer'" in "".join(cells[1]["source"])
    src = "".join(cells[2]["source"])
    assert ast.literal_eval(src.split("JOBS = ", 1)[1]) == jobs
    meta = json.loads((out / "kernel-metadata.json").read_text())
    assert meta["id"] == "someone/pod2post-07-infer" and meta["dataset_sources"] == ["someone/pod2post-adapters"]
    assert meta["enable_internet"] and len(digest) == 12


def test_unpack_adapters_rebuilds_one_folder_per_adapter(tmp_path):
    src = tmp_path / "ds"
    src.mkdir()
    for name in ("sft-dpo", "sft"):
        for f in kaggle_infer.ADAPTER_FILES:
            (src / f"{name}__{f}").write_text(name)
    paths = kaggle_jobs.unpack_adapters(src, tmp_path / "adapters")
    assert set(paths) == {"sft-dpo", "sft"}
    assert (Path(paths["sft"]) / "adapter_config.json").read_text() == "sft"
    assert kaggle_jobs.unpack_adapters(src, tmp_path / "adapters") == paths  # a re-run keeps the links


def test_write_all_uses_stage_one_transcripts_and_isolates_failures(tmp_path):
    out = tmp_path / "jobs"
    (out / "j1").mkdir(parents=True)
    (out / "j1" / "transcript.json").write_text(json.dumps({
        "utterances": [{"speaker": None, "text": "Zipline delivers blood to hospitals in Rwanda.", "start": 0.0}],
        "minutes": 8.5, "title": "Drones in Rwanda", "kind": "feed", "audio_url": "https://a/x.mp3",
        "download_seconds": 3, "transcribe_seconds": 40}))
    (out / "j2").mkdir()
    (out / "j2" / "transcript.json").write_text(json.dumps({"error": "ValueError: not a public address"}))
    jobs = [{"id": "j1", "url": "https://feed", "model": "sft_dpo"},
            {"id": "j2", "url": "http://10.0.0.1/x.mp3", "model": "sft"},
            {"id": "j3", "url": "", "transcript": TRANSCRIPT, "title": "Zipline", "model": "base"}]
    gen = SwitchingGen()
    summary = kaggle_jobs.write_all(jobs, out, {"sft-dpo": "/a/sft-dpo", "sft": "/a/sft"}, {}, gen=gen)
    assert summary["j1"]["ok"] and summary["j3"]["ok"] and "not a public address" in summary["j2"]["error"]
    assert gen.adapters == [("sft-dpo", "/a/sft-dpo"), (None, None)]
    r1 = json.loads((out / "j1" / "result.json").read_text())
    s1 = json.loads(r1["scores"])
    assert r1["title"] == "Drones in Rwanda" and s1["source"]["transcribe_seconds"] == 40
    assert s1["source"]["audio_minutes"] == 8.5 and r1["file"] == "article.md"
    assert (out / "j1" / "article.md").read_text().startswith("# Drone")
    assert json.loads((out / "j3" / "result.json").read_text())["transcript"].startswith("Ann: Zipline")


def test_uploaded_audio_attaches_the_uploads_dataset_and_is_transcribed_from_it(tmp_path, monkeypatch):
    jobs = [{"id": "j1", "url": "", "transcript": "", "audio": "abc123.mp3", "model": "sft_dpo"}]
    out, _ = kaggle_infer.build_with_jobs(jobs, "someone", out_root=tmp_path / "build")
    meta = json.loads((out / "kernel-metadata.json").read_text())
    assert meta["dataset_sources"] == ["someone/pod2post-adapters", "someone/pod2post-uploads"]

    import src.infer.article as article

    heard = []

    class FakeTranscriber:
        def __call__(self, path):
            heard.append(path)
            return [{"speaker": None, "text": "Zipline delivers blood.", "start": 0.0}], 3.0

    monkeypatch.setattr(article, "Transcriber", FakeTranscriber)
    uploads = tmp_path / "uploads"
    uploads.mkdir()
    (uploads / "abc123.mp3").write_bytes(b"audio")
    missing = {"id": "j2", "url": "", "audio": "gone.mp3"}
    kaggle_jobs.transcribe(jobs + [missing], tmp_path / "jobs", uploads)
    rec = json.loads((tmp_path / "jobs" / "j1" / "transcript.json").read_text())
    assert heard == [str(uploads / "abc123.mp3")] and rec["kind"] == "upload" and rec["minutes"] == 3.0
    assert "not in the attached dataset" in json.loads((tmp_path / "jobs" / "j2" / "transcript.json").read_text())["error"]


def test_prebuilt_env_is_used_only_if_it_imports(tmp_path, monkeypatch):
    import tarfile

    assert kaggle_jobs.use_prebuilt_env(None) is False
    (tmp_path / "vllm-env").mkdir()
    (tmp_path / "vllm-env" / "unrelated.py").write_text("")
    with tarfile.open(tmp_path / "env.tar", "w") as tar:
        tar.add(tmp_path / "vllm-env", arcname="vllm-env")
    monkeypatch.delenv("PYTHONPATH", raising=False)
    assert kaggle_jobs.use_prebuilt_env(tmp_path / "env.tar", tmp_path / "out") is False  # no vllm inside
    assert "PYTHONPATH" not in __import__("os").environ and (tmp_path / "out" / "vllm-env" / "unrelated.py").is_file()


def test_vllm_engine_gets_the_short_cuda_graph_list():
    sizes = kaggle_jobs.ENGINE["compilation_config"]["cudagraph_capture_sizes"]
    assert max(sizes) == kaggle_jobs.ENGINE["max_num_seqs"] and sizes == sorted(sizes)


def test_no_model_starts_when_every_link_failed(tmp_path):
    out = tmp_path / "jobs"
    (out / "j1").mkdir(parents=True)
    (out / "j1" / "transcript.json").write_text(json.dumps({"error": "DownloadError: Sign in to confirm"}))
    jobs = [{"id": "j1", "url": "https://www.youtube.com/watch?v=x", "model": "sft_dpo"}]
    assert not kaggle_jobs.needs_model(jobs, out)
    summary = kaggle_jobs.write_all(jobs, out, {}, {"generation": {}})  # would start vLLM if it tried
    assert "Sign in" in summary["j1"]["error"]
    assert kaggle_jobs.needs_model(jobs + [{"id": "j2", "transcript": TRANSCRIPT}], out)
