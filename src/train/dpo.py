"""Phase 6: DPO from the SFT adapter (spec §5.4) with Unsloth + TRL on one T4 (fp16).

The SFT adapter is loaded as a trainable PEFT model. TRL then copies it into a frozen
"ref" adapter, so the reference policy is the SFT model rather than the base model.
Pairs come from src.data.build_dpo.assemble(): mostly the SFT model's own less faithful
sample vs. the most faithful clean answer (D39), plus synthetic teacher-vs-corrupted pairs.
"""
from __future__ import annotations

import json
import random
import time
from pathlib import Path

from src.data.testset import HASH_FILE, assert_no_test_leak
from src.train.sft import plot_losses, time_budget_callback


def split_pairs(pairs: list[dict], seed: int, val_share: float = 0.05) -> tuple[list[dict], list[dict]]:
    eps = sorted({p["episode_id"] for p in pairs})
    val = set(random.Random(seed).sample(eps, max(1, round(len(eps) * val_share))))
    return [p for p in pairs if p["episode_id"] not in val], [p for p in pairs if p["episode_id"] in val]


def train(pairs: list[dict], data_dir: Path, sft_adapter: Path, out_dir: Path, cfg: dict, smoke: bool,
          resume_from: Path | None = None) -> dict:
    from unsloth import FastLanguageModel  # import before trl/transformers so Unsloth can patch them

    import torch
    from datasets import Dataset
    from trl import DPOConfig, DPOTrainer

    assert_no_test_leak({p["episode_id"] for p in pairs}, data_dir / HASH_FILE)
    s = cfg["smoke"]
    train_pairs, val_pairs = split_pairs(pairs, cfg["seed"])
    if smoke:
        train_pairs, val_pairs = train_pairs[: s["num_examples"]], val_pairs[:2]
    model, tok = FastLanguageModel.from_pretrained(str(sft_adapter), max_seq_length=cfg["max_length"],
                                                   dtype=torch.float16, load_in_4bit=True)
    FastLanguageModel.for_training(model)
    save_steps = s["save_steps"] if smoke else cfg["save_steps"]
    args = DPOConfig(
        output_dir=str(out_dir / "checkpoints"), beta=cfg["beta"], learning_rate=cfg["learning_rate"],
        num_train_epochs=cfg["num_train_epochs"], max_steps=s["max_steps"] if smoke else -1,
        max_length=cfg["max_length"], per_device_train_batch_size=cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=1 if smoke else cfg["gradient_accumulation_steps"],
        warmup_ratio=0.1, lr_scheduler_type="cosine", optim="adamw_8bit", fp16=True, bf16=False,
        logging_steps=1 if smoke else 5, save_steps=save_steps, save_total_limit=2,
        eval_strategy="steps", eval_steps=save_steps, per_device_eval_batch_size=1,
        seed=cfg["seed"], report_to="none", gradient_checkpointing=True)
    fields = ("prompt", "chosen", "rejected")
    budget = time_budget_callback(s["time_budget_s"] if smoke else cfg["time_budget_s"])
    trainer = DPOTrainer(model=model, ref_model=None, args=args, processing_class=tok, callbacks=[budget],
                         train_dataset=Dataset.from_list([{k: p[k] for k in fields} for p in train_pairs]),
                         eval_dataset=Dataset.from_list([{k: p[k] for k in fields} for p in val_pairs]))
    t0 = time.time()
    result = trainer.train(resume_from_checkpoint=str(resume_from) if resume_from else None)
    elapsed = time.time() - t0
    model.save_pretrained(out_dir / "adapter")
    tok.save_pretrained(out_dir / "adapter")
    history = trainer.state.log_history
    (out_dir / "log_history.json").write_text(json.dumps(history, indent=1), encoding="utf-8")
    plot_losses(history, out_dir / "loss_curve.png")
    last_eval = next((h for h in reversed(history) if "eval_loss" in h), {})
    return {"train_loss": round(result.training_loss, 4), "steps": trainer.state.global_step,
            "stopped_by_time_budget": budget.hit, "runtime_s": round(elapsed),
            "n_train": len(train_pairs), "n_val": len(val_pairs),
            "eval_reward_accuracy": last_eval.get("eval_rewards/accuracies"),
            "eval_reward_margin": last_eval.get("eval_rewards/margins"),
            "rejected_types": {t: sum(p["rejected_type"] == t for p in train_pairs)
                               for t in sorted({p["rejected_type"] for p in train_pairs})},
            "peak_gpu_mem_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2)}
