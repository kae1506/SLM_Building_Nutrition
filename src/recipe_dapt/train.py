from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

from .config import load_config
from .data import CausalLMCollator, MixedPackedDataset, evaluation_subset, load_packed_dataset
from .utils import atomic_write_json, discover_checkpoint, require_cuda_bf16, seed_everything


def main() -> None:
    parser = argparse.ArgumentParser(description="Run full-parameter Qwen recipe DAPT")
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", action="store_true", help="Resume the newest checkpoint")
    parser.add_argument("--force", action="store_true", help="Retrain even if final/ exists")
    args = parser.parse_args()

    config = load_config(args.config)
    require_cuda_bf16()
    seed = int(config.data["seed"])
    seed_everything(seed)
    output_dir = config.output_dir
    final_dir = output_dir / "final"
    if final_dir.is_dir() and not args.force:
        print(f"Final model already exists; skipping training: {final_dir}")
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output_dir / "resolved_config.json", config.resolved_dict())

    max_steps = config.max_steps()
    total_examples = max_steps * int(config.training["global_batch_sequences"])
    recipe = load_packed_dataset(
        config.paths["recipe_tokens"], "train", seed=seed
    )
    replay_ratio = float(config.data["replay_ratio"])
    replay = None
    if replay_ratio:
        replay = load_packed_dataset(
            config.paths["replay_tokens"], "train", seed=seed + 1
        )
    train_dataset = MixedPackedDataset(
        recipe=recipe,
        replay=replay,
        replay_ratio=replay_ratio,
        total_examples=total_examples,
    )
    eval_dataset = evaluation_subset(
        config.paths["recipe_tokens"],
        "validation",
        int(config.training["eval_blocks"]),
        seed + 2,
    )

    attention = str(config.training["attention_implementation"])
    try:
        model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            torch_dtype=torch.bfloat16,
            attn_implementation=attention,
        )
    except Exception as exc:
        if attention == "flash_attention_2":
            raise RuntimeError(
                "FlashAttention 2 model loading failed. Install flash-attn or set "
                "DAPT_ATTENTION=sdpa."
            ) from exc
        raise
    model.config.use_cache = False
    if bool(config.training["gradient_checkpointing"]):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    gradient_accumulation = config.gradient_accumulation_steps(world_size)
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        max_steps=max_steps,
        per_device_train_batch_size=int(config.training["per_device_train_batch_size"]),
        per_device_eval_batch_size=int(config.training["per_device_eval_batch_size"]),
        gradient_accumulation_steps=gradient_accumulation,
        learning_rate=float(config.training["learning_rate"]),
        lr_scheduler_type="cosine",
        warmup_ratio=float(config.training["warmup_ratio"]),
        weight_decay=float(config.training["weight_decay"]),
        adam_beta1=float(config.training["adam_beta1"]),
        adam_beta2=float(config.training["adam_beta2"]),
        adam_epsilon=float(config.training["adam_epsilon"]),
        max_grad_norm=float(config.training["max_grad_norm"]),
        bf16=True,
        bf16_full_eval=True,
        tf32=True,
        optim="adamw_torch_fused",
        eval_strategy="steps",
        eval_steps=int(config.training["eval_steps"]),
        save_strategy="steps",
        save_steps=int(config.training["save_steps"]),
        logging_strategy="steps",
        logging_steps=int(config.training["logging_steps"]),
        save_total_limit=int(config.training["save_total_limit"]),
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to=["tensorboard"],
        run_name=config.run_name,
        seed=seed,
        data_seed=seed,
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        dataloader_persistent_workers=True,
        remove_unused_columns=False,
        torch_compile=bool(config.training["torch_compile"]),
        ddp_find_unused_parameters=False,
    )
    tokenizer = AutoTokenizer.from_pretrained(config.model_name, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=CausalLMCollator(),
        processing_class=tokenizer,
    )

    checkpoint = None
    if args.resume or os.environ.get("DAPT_RESUME", "1").lower() not in {"0", "false", "no"}:
        checkpoint = discover_checkpoint(output_dir)
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    result = trainer.train(resume_from_checkpoint=checkpoint)
    elapsed = time.monotonic() - started
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(final_dir)
    trainer.save_state()

    metrics = dict(result.metrics)
    trained_tokens = max_steps * config.tokens_per_step()
    metrics.update(
        {
            "run_name": config.run_name,
            "model_name": config.model_name,
            "max_steps": max_steps,
            "trained_tokens": trained_tokens,
            "tokens_per_step": config.tokens_per_step(),
            "gradient_accumulation_steps": gradient_accumulation,
            "replay_ratio": replay_ratio,
            "wall_seconds": elapsed,
            "tokens_per_second": trained_tokens / elapsed,
            "peak_gpu_memory_gib": torch.cuda.max_memory_allocated() / (1024**3),
            "best_checkpoint": trainer.state.best_model_checkpoint,
            "best_metric": trainer.state.best_metric,
            "final_model": str(final_dir),
        }
    )
    atomic_write_json(output_dir / "training_summary.json", metrics)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
