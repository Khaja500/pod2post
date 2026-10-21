"""Phase 3 teacher generation (kernel 02_teacher_gen, teacher option 1).

Qwen3-14B-AWQ on vLLM across both T4s (tensor parallel), thinking disabled:
  1. Stage A: notes for every chunk of the selected episodes (one retry on invalid JSON)
  2. merge notes per episode (<= 2,500 tokens) and pick the target keywords
  3. Stage B: article + social pack
  4. "generic" rewrites of some Stage B outputs (DPO rejected type (d), spec §4.4)

Resumable: records are appended to /kaggle/working/teacher/*.jsonl as each batch
finishes; a rerun with the previous output attached copies them and skips what is
done. New batches stop after `time_budget_s`, so the run ends inside the 3 h limit.
"""
from __future__ import annotations

import json
import random
import shutil
import time
from collections import defaultdict
from pathlib import Path

from src.data.chunk import read_jsonl
from src.data.notes import merge_notes, parse_notes, pick_keywords, sanitize_notes
from src.data.prompts import stage_a_messages, stage_b_messages
from src.data.testset import HASH_FILE, assert_no_test_leak
from src.eval.seo import format_problems, parse_output

GENERIC_SYSTEM = "You are an editor. You rewrite content exactly as instructed and keep its format."
GENERIC_USER = """Rewrite the content below so that it becomes generic and vague: remove the specific names, \
numbers, quotes and concrete examples and replace them with general statements that could apply to any \
episode on the topic. Keep exactly the same format: <article> with the title line, the "Meta description:" \
line, the H2 sections and the FAQ, then <social> with the same JSON keys.

{output}"""


def select_episodes(chunks: list[dict], cfg: dict, smoke: bool, seed: int) -> list[str]:
    """Stage B episodes: a share of long ones (many chunks, like the test set) + short ones."""
    n_chunks = {c["episode_id"]: c["n_chunks"] for c in chunks if c["split"] == "train"}
    long = sorted(e for e, n in n_chunks.items() if n >= cfg["long_min_chunks"])
    short = sorted(e for e, n in n_chunks.items() if n < cfg["long_min_chunks"])
    rng = random.Random(seed)
    if smoke:
        return rng.sample(short, 2) + rng.sample(long, 1)
    total = round(cfg["targets"]["stage_b"] * (1 + cfg["overgenerate"]))
    n_long = min(len(long), round(total * cfg["long_share"]))
    return rng.sample(long, n_long) + rng.sample(short, total - n_long)


def Teacher(cfg: dict):
    """The teacher model as a shared vLLM generator (tests replace this with a fake)."""
    from src.infer.generator import VLLMGenerator

    m = cfg["open_model"]
    return VLLMGenerator(m["model"], tensor_parallel_size=m["tensor_parallel_size"], max_model_len=m["max_model_len"],
                         gpu_memory_utilization=m["gpu_memory_utilization"], seed=cfg["seed"],
                         temperature=m["temperature"], top_p=m["top_p"], top_k=m["top_k"])


def _load(path: Path) -> list[dict]:
    return list(read_jsonl(path)) if path.is_file() else []


def _append(path: Path, rows: list[dict]) -> None:
    with path.open("a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _n_problems(text: str) -> int:
    return len(format_problems(*parse_output(text, repair=True)))


def chat_with_format_retry(teacher, convs: list, max_tokens: int) -> tuple[list[str], list[bool]]:
    """Generate, then retry once (temperature 0.5) the outputs that break a hard format rule;
    keep whichever attempt breaks fewer rules. Data generation only; evaluation is single-shot."""
    texts = teacher.chat(convs, max_tokens)
    retry = [i for i, t in enumerate(texts) if _n_problems(t)]
    if retry:
        for i, t in zip(retry, teacher.chat([convs[i] for i in retry], max_tokens, temperature=0.5)):
            if _n_problems(t) < _n_problems(texts[i]):
                texts[i] = t
    return texts, [i in retry for i in range(len(texts))]


def _batches(items: list, size: int):
    for i in range(0, len(items), size):
        yield items[i:i + size]


def run(data_dir: Path, out_dir: Path, cfg: dict, smoke: bool, resume_dir: Path | None = None) -> dict:
    t0 = time.time()
    budget = cfg["smoke_time_budget_s"] if smoke else cfg["time_budget_s"]
    out_dir.mkdir(parents=True, exist_ok=True)
    if resume_dir is not None:
        for f in ("stage_a.jsonl", "stage_b.jsonl", "generic.jsonl"):
            if (resume_dir / f).is_file() and not (out_dir / f).is_file():
                shutil.copy(resume_dir / f, out_dir / f)

    chunks = [c for c in read_jsonl(data_dir / "chunks.jsonl") if c["split"] == "train"]
    titles = {e["episode_id"]: e["title"] for e in read_jsonl(data_dir / "episodes.jsonl") if e["split"] == "train"}
    selected = select_episodes(chunks, cfg, smoke, cfg["seed"])
    assert_no_test_leak(selected, data_dir / HASH_FILE)
    by_ep = defaultdict(list)
    for c in chunks:
        by_ep[c["episode_id"]].append(c)
    teacher = Teacher(cfg)
    stats = {"selected_episodes": len(selected), "stopped_early": False}

    def out_of_time() -> bool:
        stats["stopped_early"] = stats["stopped_early"] or time.time() - t0 > budget
        return stats["stopped_early"]

    # 1. Stage A over every chunk of the selected episodes.
    a_path = out_dir / "stage_a.jsonl"
    done_a = {(r["episode_id"], r["chunk_idx"]) for r in _load(a_path)}
    todo = [c for e in selected for c in by_ep[e] if (c["episode_id"], c["chunk_idx"]) not in done_a]
    for batch in _batches(todo, cfg["batch_size"]):
        if out_of_time():
            break
        convs = [stage_a_messages(titles[c["episode_id"]], c["text"], c["chunk_idx"] + 1, c["n_chunks"]) for c in batch]
        texts = teacher.chat(convs, cfg["max_tokens"]["stage_a"])
        notes = [parse_notes(t) for t in texts]
        retry = [i for i, n in enumerate(notes) if n is None]
        if retry:
            again = teacher.chat([convs[i] for i in retry], cfg["max_tokens"]["stage_a"], temperature=0.3)
            for i, t in zip(retry, again):
                texts[i], notes[i] = t, parse_notes(t)
        _append(a_path, [{"episode_id": c["episode_id"], "chunk_idx": c["chunk_idx"], "n_chunks": c["n_chunks"],
                          "output": t, "notes": n, "retried": i in retry}
                         for i, (c, t, n) in enumerate(zip(batch, texts, notes))])
        print(f"[teacher] stage A {len(_load(a_path))} chunks, {time.time() - t0:.0f}s", flush=True)

    # 2 + 3. Merge notes for complete episodes, then Stage B.
    notes_by_ep = defaultdict(dict)
    for r in _load(a_path):
        notes_by_ep[r["episode_id"]][r["chunk_idx"]] = r["notes"]
    b_path = out_dir / "stage_b.jsonl"
    done_b = {r["episode_id"] for r in _load(b_path)}
    ready = []
    for e in selected:
        got = notes_by_ep.get(e, {})
        if e in done_b or len(got) < len(by_ep[e]) or any(n is None for n in got.values()):
            continue  # unfinished, or a chunk whose notes never parsed
        text_of = {c["chunk_idx"]: c["text"] for c in by_ep[e]}
        ordered = [sanitize_notes(got[i], text_of[i]) for i in sorted(got)]
        primary, secondary = pick_keywords(ordered)
        if primary:
            ready.append({"episode_id": e, "title": titles[e], "primary_keyword": primary,
                          "secondary_keywords": secondary,
                          "notes_text": merge_notes(ordered, teacher.count, cfg["notes_budget_tokens"])})
    for batch in _batches(ready, cfg["batch_size"]):
        if out_of_time():
            break
        texts, retried = chat_with_format_retry(teacher, [stage_b_messages(
            r["title"], r["notes_text"], r["primary_keyword"], r["secondary_keywords"]) for r in batch],
            cfg["max_tokens"]["stage_b"])
        _append(b_path, [{**r, "output": t, "retried": f} for r, t, f in zip(batch, texts, retried)])
        print(f"[teacher] stage B {len(_load(b_path))} episodes, {time.time() - t0:.0f}s", flush=True)

    # 4. Generic rewrites for DPO.
    g_path = out_dir / "generic.jsonl"
    done_g = {r["episode_id"] for r in _load(g_path)}
    n_generic = 1 if smoke else cfg["generic_rewrites"]
    sources = [r for r in _load(b_path) if "<article>" in r["output"] and r["episode_id"] not in done_g]
    sources = sources[:max(0, n_generic - len(done_g))]
    for batch in _batches(sources, cfg["batch_size"]):
        if out_of_time():
            break
        texts, _ = chat_with_format_retry(teacher, [[{"role": "system", "content": GENERIC_SYSTEM},
                                                     {"role": "user", "content": GENERIC_USER.format(output=r["output"])}]
                                                    for r in batch], cfg["max_tokens"]["stage_b"])
        _append(g_path, [{"episode_id": r["episode_id"], "output": t} for r, t in zip(batch, texts)])

    elapsed = time.time() - t0
    stats.update(stage_a=len(_load(a_path)), stage_a_invalid=sum(r["notes"] is None for r in _load(a_path)),
                 stage_b=len(_load(b_path)), generic=len(_load(g_path)), elapsed_s=round(elapsed),
                 generated_tokens=teacher.generated_tokens,
                 tokens_per_s=round(teacher.generated_tokens / max(elapsed, 1), 1), model=cfg["open_model"]["model"])
    (out_dir / "teacher_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    return stats
