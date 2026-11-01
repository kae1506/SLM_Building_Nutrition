from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import load_config
from .data import CausalLMCollator, evaluation_subset
from .utils import atomic_write_json, require_cuda_bf16, seed_everything


def evaluate(model_path: str, dataset_root: Path, split: str, blocks: int, batch_size: int, seed: int) -> dict:
    require_cuda_bf16()
    seed_everything(seed)
    dataset = evaluation_subset(dataset_root, split, blocks, seed)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        collate_fn=CausalLMCollator(),
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    ).cuda()
    model.eval()
    torch.cuda.reset_peak_memory_stats()
    total_nll = 0.0
    total_tokens = 0
    correct = 0
    started = time.monotonic()
    with torch.inference_mode():
        for batch in loader:
            batch = {key: value.cuda(non_blocking=True) for key, value in batch.items()}
            outputs = model(**batch)
            predicted = outputs.logits[:, :-1].argmax(dim=-1)
            labels = batch["labels"][:, 1:]
            token_count = labels.numel()
            total_nll += float(outputs.loss) * token_count
            total_tokens += token_count
            correct += int((predicted == labels).sum())
    elapsed = time.monotonic() - started
    mean_loss = total_nll / total_tokens
    metrics = {
        "model": model_path,
        "dataset": str(dataset_root),
        "split": split,
        "blocks": len(dataset),
        "tokens": total_tokens,
        "loss": mean_loss,
        "perplexity": math.exp(min(mean_loss, 20.0)),
        "next_token_accuracy": correct / total_tokens,
        "elapsed_seconds": elapsed,
        "tokens_per_second": total_tokens / elapsed,
        "peak_gpu_memory_gib": torch.cuda.max_memory_allocated() / (1024**3),
    }
    del model
    torch.cuda.empty_cache()
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure held-out causal-LM perplexity and token accuracy")
    parser.add_argument("--config", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", choices=["recipe", "replay"], default="recipe")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--blocks", type=int)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    root = config.paths["recipe_tokens"] if args.dataset == "recipe" else config.paths["replay_tokens"]
    blocks = args.blocks or int(config.training["eval_blocks"])
    metrics = evaluate(
        model_path=args.model,
        dataset_root=root,
        split=args.split,
        blocks=blocks,
        batch_size=int(config.training["per_device_eval_batch_size"]),
        seed=int(config.data["seed"]) + 17,
    )
    atomic_write_json(args.output, metrics)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
