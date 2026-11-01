from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import load_config
from .utils import atomic_write_json, require_cuda_bf16, seed_everything


def load_recipe_rows(root: Path, limit: int, seed: int) -> list[dict[str, Any]]:
    """Deterministically reservoir-sample eligible rows across the entire test split."""
    rng = random.Random(seed)
    rows: list[dict[str, Any]] = []
    eligible = 0
    columns = ["title", "ingredients", "instructions"]
    for path in sorted((root / "test").glob("*.parquet")):
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(batch_size=1024, columns=columns):
            for row in batch.to_pylist():
                if row["title"] and len(row["ingredients"] or []) >= 2 and len(row["instructions"] or []) >= 2:
                    eligible += 1
                    if len(rows) < limit:
                        rows.append(row)
                    else:
                        replacement = rng.randrange(eligible)
                        if replacement < limit:
                            rows[replacement] = row
    return rows


def make_examples(rows: list[dict[str, Any]], per_task: int, seed: int) -> dict[str, list[dict]]:
    rng = random.Random(seed)
    rng.shuffle(rows)
    ingredient_pool = [item for row in rows for item in row["ingredients"] if 2 <= len(item) <= 120]
    instruction_pool = [item for row in rows for item in row["instructions"] if 8 <= len(item) <= 240]
    ingredient_examples: list[dict] = []
    next_step_examples: list[dict] = []

    for row in rows:
        if len(ingredient_examples) < per_task:
            target_index = rng.randrange(min(3, len(row["ingredients"])))
            correct = row["ingredients"][target_index]
            previous = row["ingredients"][:target_index]
            distractors = _draw_distractors(ingredient_pool, correct, set(row["ingredients"]), rng)
            if len(distractors) == 3:
                options = [correct, *distractors]
                rng.shuffle(options)
                context = f"Recipe: {row['title']}\n\nIngredients:\n"
                if previous:
                    context += "".join(f"- {item}\n" for item in previous)
                context += "-"
                ingredient_examples.append(
                    {"context": context, "options": [" " + item for item in options], "answer": options.index(correct)}
                )

        if len(next_step_examples) < per_task:
            target_index = rng.randrange(1, min(len(row["instructions"]), 6))
            correct = row["instructions"][target_index]
            distractors = _draw_distractors(instruction_pool, correct, set(row["instructions"]), rng)
            if len(distractors) == 3:
                options = [correct, *distractors]
                rng.shuffle(options)
                ingredients = "".join(f"- {item}\n" for item in row["ingredients"])
                steps = "".join(
                    f"{index + 1}. {step}\n"
                    for index, step in enumerate(row["instructions"][:target_index])
                )
                context = (
                    f"Recipe: {row['title']}\n\nIngredients:\n{ingredients}\n"
                    f"Instructions:\n{steps}{target_index + 1}."
                )
                next_step_examples.append(
                    {"context": context, "options": [" " + item for item in options], "answer": options.index(correct)}
                )
        if len(ingredient_examples) >= per_task and len(next_step_examples) >= per_task:
            break
    return {"ingredient_cloze": ingredient_examples, "next_step": next_step_examples}


def _draw_distractors(pool: list[str], correct: str, excluded: set[str], rng: random.Random) -> list[str]:
    selected: list[str] = []
    for _ in range(100):
        candidate = rng.choice(pool)
        if candidate != correct and candidate not in excluded and candidate not in selected:
            selected.append(candidate)
            if len(selected) == 3:
                break
    return selected


def score_options(model, tokenizer, context: str, options: list[str], max_length: int = 1024) -> list[float]:
    context_ids = tokenizer(context, add_special_tokens=False)["input_ids"]
    option_ids = [tokenizer(option, add_special_tokens=False)["input_ids"] for option in options]
    if not context_ids or any(not ids for ids in option_ids):
        raise ValueError("Benchmark context and options must each contain at least one token")
    max_option = max(len(ids) for ids in option_ids)
    context_ids = context_ids[-(max_length - max_option) :]
    sequences = [context_ids + ids for ids in option_ids]
    lengths = [len(sequence) for sequence in sequences]
    pad_id = tokenizer.pad_token_id
    input_ids = torch.full((len(sequences), max(lengths)), pad_id, dtype=torch.long)
    attention = torch.zeros_like(input_ids)
    for index, sequence in enumerate(sequences):
        input_ids[index, : len(sequence)] = torch.tensor(sequence)
        attention[index, : len(sequence)] = 1
    input_ids = input_ids.cuda(non_blocking=True)
    attention = attention.cuda(non_blocking=True)
    with torch.inference_mode():
        logits = model(input_ids=input_ids, attention_mask=attention).logits
    scores: list[float] = []
    context_length = len(context_ids)
    for row, ids in enumerate(option_ids):
        prediction_logits = logits[row, context_length - 1 : context_length + len(ids) - 1]
        targets = input_ids[row, context_length : context_length + len(ids)]
        losses = F.cross_entropy(prediction_logits.float(), targets, reduction="none")
        scores.append(-float(losses.mean()))
    return scores


def evaluate_examples(model, tokenizer, examples: list[dict]) -> dict:
    if not examples:
        raise RuntimeError("No valid held-out benchmark examples could be constructed")
    correct = 0
    margins: list[float] = []
    for index, example in enumerate(examples, start=1):
        scores = score_options(model, tokenizer, example["context"], example["options"])
        prediction = max(range(len(scores)), key=scores.__getitem__)
        correct += int(prediction == example["answer"])
        correct_score = scores[example["answer"]]
        best_wrong = max(score for choice, score in enumerate(scores) if choice != example["answer"])
        margins.append(correct_score - best_wrong)
        if index % 25 == 0:
            print(f"domain benchmark: {index}/{len(examples)}", flush=True)
    return {
        "examples": len(examples),
        "accuracy": correct / len(examples),
        "mean_correct_logprob_margin": sum(margins) / len(margins),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Likelihood-based recipe knowledge benchmarks")
    parser.add_argument("--config", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    require_cuda_bf16()
    seed = int(config.data["seed"]) + 31
    seed_everything(seed)
    per_task = int(config.benchmarks["recipe_examples_per_task"])
    rows = load_recipe_rows(
        config.paths["recipe_source"], max(per_task * 20, 3000), seed
    )
    examples = make_examples(rows, per_task, seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    ).cuda().eval()
    started = time.monotonic()
    results = {
        "model": args.model,
        "benchmark": "held-out Recipe1M + RecipeNLG likelihood ranking",
        "tasks": {
            task: evaluate_examples(model, tokenizer, task_examples)
            for task, task_examples in examples.items()
        },
    }
    results["mean_accuracy"] = sum(item["accuracy"] for item in results["tasks"].values()) / len(results["tasks"])
    results["elapsed_seconds"] = time.monotonic() - started
    atomic_write_json(args.output, results)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
