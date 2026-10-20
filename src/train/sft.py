"""Phase 5: QLoRA SFT with Unsloth + TRL on one T4 (fp16; spec §5.3). One adapter, both tasks.

Data: data/processed/sft.jsonl (src.data.build_sft) as conversational prompt-completion
pairs, with the loss on the completion only. Checkpoints go to out_dir/checkpoints every
save_steps. A rerun resumes from the newest checkpoint of an attached previous run, and
training saves and stops itself once the time budget is spent.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from src.data.chunk import read_jsonl
from src.data.testset import HASH_FILE, assert_no_test_leak


def load_rows(data_dir: Path, smoke: bool, n_smoke: int) -> tuple[list[dict], list[dict]]:
    rows = list(read_jsonl(data_dir / "sft.jsonl"))
    assert_no_test_leak({r["episode_id"] for r in rows}, data_dir / HASH_FILE)
    train = [r for r in rows if r["split"] == "train"]
    val = [r for r in rows if r["split"] == "val"]
    if smoke:  # both tasks in the smoke sample
        train = [r for r in train if r["task"] == "A"][: n_smoke // 2] + [r for r in train if r["task"] == "B"][: n_smoke - n_smoke // 2]
        val = val[:4]
    return train, val


def prompt_completion(rows: list[dict]) -> list[dict]:
    return [{"prompt": r["messages"][:-1], "completion": r["messages"][-1:]} for r in rows]


def time_budget_callback(budget_s: float):
    from transformers import TrainerCallback

    class TimeBudget(TrainerCallback):
        def __init__(self):
            self.t0 = time.time()
            self.hit = False

        def on_step_end(self, args, state, control, **kwargs):
            if time.time() - self.t0 > budget_s:
                self.hit = True
                control.should_save = True
                control.should_training_stop = True
            return control

    return TimeBudget()


def plot_losses(history: list[dict], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    train = [(h["step"], h["loss"]) for h in history if "loss" in h]
    evals = [(h["step"], h["eval_loss"]) for h in history if "eval_loss" in h]
    fig, ax = plt.subplots(figsize=(7, 4))
    if train:
        ax.plot(*zip(*train), label="train loss")
    if evals:
        ax.plot(*zip(*evals), "o-", label="val loss")
    ax.set_xlabel("step")
    ax.set_ylabel("loss")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)


def train(data_dir: Path, out_dir: Path, cfg: dict, smoke: bool, resume_from: Path | None = None) -> dict:
    from unsloth import FastLanguageModel  # import before trl/transformers so Unsloth can patch them

    import torch
    from datasets import Dataset
    from trl import SFTConfig, SFTTrainer

    s = cfg["smoke"]
    train_rows, val_rows = load_rows(data_dir, smoke, s["num_examples"])
    model, tok = FastLanguageModel.from_pretrained(cfg["base_model"], max_seq_length=cfg["max_seq_length"],
                                                   dtype=torch.float16, load_in_4bit=cfg["load_in_4bit"])
    model = FastLanguageModel.get_peft_model(
        model, r=cfg["lora_r"], lora_alpha=cfg["lora_alpha"], lora_dropout=cfg["lora_dropout"],
        target_modules=cfg["target_modules"], use_gradient_checkpointing="unsloth", random_state=cfg["seed"])
    save_steps = s["save_steps"] if smoke else cfg["save_steps"]
    args = SFTConfig(
        output_dir=str(out_dir / "checkpoints"), per_device_train_batch_size=cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=1 if smoke else cfg["gradient_accumulation_steps"],
        learning_rate=cfg["learning_rate"], num_train_epochs=cfg["num_train_epochs"],
        max_steps=s["max_steps"] if smoke else -1, warmup_ratio=cfg["warmup_ratio"],
        lr_scheduler_type=cfg["lr_scheduler"], weight_decay=0.01, optim="adamw_8bit",
        fp16=True, bf16=False, max_length=cfg["max_seq_length"], completion_only_loss=True,
        logging_steps=1 if smoke else 5, save_steps=save_steps, save_total_limit=2,
        eval_strategy="steps", eval_steps=save_steps, per_device_eval_batch_size=1,
        seed=cfg["seed"], report_to="none", dataset_num_proc=2)
    budget = time_budget_callback(s["time_budget_s"] if smoke else cfg["time_budget_s"])
    trainer = SFTTrainer(model=model, processing_class=tok, args=args,
                         train_dataset=Dataset.from_list(prompt_completion(train_rows)),
                         eval_dataset=Dataset.from_list(prompt_completion(val_rows)), callbacks=[budget])
    t0 = time.time()
    result = trainer.train(resume_from_checkpoint=str(resume_from) if resume_from else None)
    elapsed = time.time() - t0
    model.save_pretrained(out_dir / "adapter")
    tok.save_pretrained(out_dir / "adapter")
    history = trainer.state.log_history
    (out_dir / "log_history.json").write_text(json.dumps(history, indent=1), encoding="utf-8")
    plot_losses(history, out_dir / "loss_curve.png")
    evals = [h["eval_loss"] for h in history if "eval_loss" in h]
    n_tokens = sum(len(tok.apply_chat_template(r["messages"], tokenize=True)) for r in train_rows)
    return {"train_loss": round(result.training_loss, 4), "final_eval_loss": evals[-1] if evals else None,
            "first_eval_loss": evals[0] if evals else None, "steps": trainer.state.global_step,
            "max_steps": trainer.state.max_steps, "stopped_by_time_budget": budget.hit,
            "runtime_s": round(elapsed), "n_train": len(train_rows), "n_val": len(val_rows),
            "train_tokens_per_epoch": n_tokens, "resumed_from": str(resume_from) if resume_from else None,
            "peak_gpu_mem_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2)}
