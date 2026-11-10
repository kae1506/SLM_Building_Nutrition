from __future__ import annotations

import argparse
import gc
import json
import math
import os
import time
from pathlib import Path
from typing import Any

import torch
from olm.data.datasets import DataLoader
from olm.train.callbacks import CheckpointCallback, ValidationCallback
from olm.train.optim import AdamW
from olm.train.schedulers import WarmupCosineScheduler
from olm.train.trainer import Trainer, TrainerCallback
from torch.utils.tensorboard import SummaryWriter

from .config import load_config
from .data import MixedPackedDataset, OLMPackedDataset, evaluation_subset, load_packed_dataset
from .olm_model import (
    CausalNextTokenLoss,
    OLM_GIT_COMMIT,
    export_olm_to_hf,
    load_pretrained_olm_qwen,
    model_state_to_cpu,
)
from .utils import atomic_write_json, discover_olm_checkpoint, require_cuda_bf16, seed_everything


class TensorBoardMetricsCallback(TrainerCallback):
    def __init__(self, log_dir: Path) -> None:
        self.writer = SummaryWriter(log_dir=str(log_dir))

    def on_step_end(self, trainer: Trainer, step: int, loss: float) -> None:
        self.writer.add_scalar("train/loss", loss, step)
        self.writer.add_scalar("train/perplexity", math.exp(min(loss, 20.0)), step)
        self.writer.add_scalar("train/learning_rate", trainer.optimizer.param_groups[0]["lr"], step)
        if "val_loss" in trainer.training_state:
            self.writer.add_scalar("eval/loss", trainer.training_state["val_loss"], step)

    def on_train_end(self, trainer: Trainer) -> None:
        self.writer.flush()
        self.writer.close()


def optimizer_groups(model: torch.nn.Module, weight_decay: float) -> list[dict[str, Any]]:
    decay: list[torch.nn.Parameter] = []
    no_decay: list[torch.nn.Parameter] = []
    for parameter in model.parameters():
        if parameter.requires_grad:
            (decay if parameter.ndim >= 2 else no_decay).append(parameter)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def load_training_checkpoint(path: Path, trainer: Trainer, device: torch.device) -> None:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    trainer.model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    trainer.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if checkpoint.get("scheduler_state_dict") is not None:
        trainer.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    if checkpoint.get("scaler_state_dict"):
        trainer.scaler.load_state_dict(checkpoint["scaler_state_dict"])
    trainer.global_step = int(checkpoint.get("step", 0))
    trainer.current_epoch = int(checkpoint.get("epoch", 0))
    trainer.losses = list(checkpoint.get("losses", []))


def save_final_training_state(path: Path, trainer: Trainer) -> None:
    payload = {
        "step": trainer.global_step,
        "epoch": trainer.current_epoch,
        "model_state_dict": trainer.model.state_dict(),
        "optimizer_state_dict": trainer.optimizer.state_dict(),
        "scheduler_state_dict": trainer.scheduler.state_dict(),
        "scaler_state_dict": trainer.scaler.state_dict(),
        "losses": trainer.losses,
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Qwen recipe DAPT with OLM")
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", action="store_true", help="Resume newest OLM checkpoint")
    parser.add_argument("--force", action="store_true", help="Retrain even if final/ exists")
    args = parser.parse_args()

    config = load_config(args.config)
    require_cuda_bf16()
    seed = int(config.data["seed"])
    seed_everything(seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    output_dir = config.output_dir
    final_dir = output_dir / "final"
    final_olm_dir = output_dir / "final_olm"
    checkpoint_dir = output_dir / "checkpoints"
    if final_dir.is_dir() and not args.force:
        print(f"Final model already exists; skipping training: {final_dir}")
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    final_olm_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output_dir / "resolved_config.json", config.resolved_dict())

    max_steps = config.max_steps()
    global_batch = int(config.training["global_batch_sequences"])
    total_examples = max_steps * global_batch
    replay_ratio = float(config.data["replay_ratio"])
    recipe = load_packed_dataset(config.paths["recipe_tokens"], "train", seed=seed)
    replay = None
    if replay_ratio:
        replay = load_packed_dataset(config.paths["replay_tokens"], "train", seed=seed + 1)
    mixed = MixedPackedDataset(recipe, replay, replay_ratio, total_examples)

    resume_enabled = args.resume or os.environ.get("DAPT_RESUME", "1").lower() not in {
        "0", "false", "no"
    }
    latest = discover_olm_checkpoint(checkpoint_dir) if resume_enabled else None
    resume_step = latest[0] if latest else 0
    if resume_step > max_steps:
        raise RuntimeError(f"Checkpoint step {resume_step} exceeds max_steps {max_steps}")
    train_dataset = OLMPackedDataset(mixed, start_index=resume_step * global_batch)
    eval_dataset = OLMPackedDataset(
        evaluation_subset(
            config.paths["recipe_tokens"],
            "validation",
            int(config.training["eval_blocks"]),
            seed + 2,
        )
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(config.training["per_device_train_batch_size"]),
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=int(config.training["per_device_eval_batch_size"]),
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    print(f"Loading {config.model_name} into OLM Qwen2Model", flush=True)
    model, hf_config = load_pretrained_olm_qwen(config.model_name, torch.bfloat16)
    device = torch.device("cuda:0")
    model = model.to(device)
    if bool(config.training.get("torch_compile", False)):
        model = torch.compile(model)
    if bool(config.training.get("gradient_checkpointing", False)):
        raise ValueError("OLM Qwen2 does not currently expose gradient checkpointing")

    optimizer = AdamW(
        optimizer_groups(model, float(config.training["weight_decay"])),
        lr=float(config.training["learning_rate"]),
        betas=(float(config.training["adam_beta1"]), float(config.training["adam_beta2"])),
        eps=float(config.training["adam_epsilon"]),
        fused=True,
    )
    warmup_steps = math.ceil(max_steps * float(config.training["warmup_ratio"]))
    scheduler = WarmupCosineScheduler(
        optimizer, warmup_steps=warmup_steps, total_steps=max_steps, min_lr=0.0
    )
    validation = ValidationCallback(
        eval_loader,
        eval_every=int(config.training["eval_steps"]),
        device=str(device),
        use_amp=False,
    )
    checkpoints = CheckpointCallback(
        checkpoint_dir=str(checkpoint_dir),
        save_every=int(config.training["save_steps"]),
        keep_last_n=int(config.training["save_total_limit"]),
        save_best=True,
    )
    metrics_logger = TensorBoardMetricsCallback(output_dir / "runs")
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        dataloader=train_loader,
        device=str(device),
        context_length=int(config.data["sequence_length"]),
        grad_accum_steps=config.gradient_accumulation_steps(world_size=1),
        use_amp=False,
        loss=CausalNextTokenLoss,
        callbacks=[validation, checkpoints, metrics_logger],
        scheduler=scheduler,
        grad_clip_norm=float(config.training["max_grad_norm"]),
        warmup_steps=warmup_steps,
        total_steps=max_steps,
        learning_rate=float(config.training["learning_rate"]),
        weight_decay=float(config.training["weight_decay"]),
    )
    if latest:
        print(f"Resuming OLM checkpoint: {latest[1]}", flush=True)
        load_training_checkpoint(latest[1], trainer, device)

    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    initial_step = trainer.global_step
    if initial_step < max_steps:
        trainer.train(
            epochs=1,
            max_steps=max_steps,
            log_interval=int(config.training["logging_steps"]),
        )
    elapsed = time.monotonic() - started
    peak_gpu_memory = torch.cuda.max_memory_allocated() / (1024**3)
    save_final_training_state(output_dir / "final_training_state.pt", trainer)

    best_checkpoint = checkpoint_dir / "best_model.pt"
    best_metric = validation.best_val_loss
    if best_checkpoint.is_file():
        best = torch.load(best_checkpoint, map_location=device, weights_only=False)
        trainer.model.load_state_dict(best["model_state_dict"], strict=True)
        best_step = int(best.get("step", trainer.global_step))
    else:
        best_step = trainer.global_step

    olm_state = model_state_to_cpu(trainer.model)
    torch.save(olm_state, final_olm_dir / "model_state.pt")
    atomic_write_json(
        final_olm_dir / "metadata.json",
        {
            "format": "OLM Qwen2Model state_dict",
            "olm_git_commit": OLM_GIT_COMMIT,
            "source_checkpoint": config.model_name,
            "selected_step": best_step,
            "dtype": "bfloat16",
        },
    )
    del trainer, optimizer, scheduler, model
    torch.cuda.empty_cache()
    gc.collect()
    export_olm_to_hf(olm_state, hf_config, config.model_name, final_dir)

    session_steps = max_steps - initial_step
    session_tokens = session_steps * config.tokens_per_step()
    metrics = {
        "run_name": config.run_name,
        "model_name": config.model_name,
        "training_library": "openlanguagemodel",
        "olm_git_commit": OLM_GIT_COMMIT,
        "max_steps": max_steps,
        "resumed_from_step": initial_step,
        "trained_tokens": max_steps * config.tokens_per_step(),
        "session_trained_tokens": session_tokens,
        "tokens_per_step": config.tokens_per_step(),
        "gradient_accumulation_steps": config.gradient_accumulation_steps(world_size=1),
        "replay_ratio": replay_ratio,
        "wall_seconds": elapsed,
        "tokens_per_second": session_tokens / elapsed if elapsed else None,
        "peak_gpu_memory_gib": peak_gpu_memory,
        "best_checkpoint": str(best_checkpoint) if best_checkpoint.is_file() else None,
        "best_step": best_step,
        "best_metric": best_metric if math.isfinite(best_metric) else None,
        "final_olm_model": str(final_olm_dir),
        "final_model": str(final_dir),
    }
    atomic_write_json(output_dir / "training_summary.json", metrics)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
