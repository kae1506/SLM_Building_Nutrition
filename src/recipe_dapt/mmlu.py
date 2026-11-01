from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from .utils import atomic_write_json


def run_mmlu(model: str, output_dir: Path, num_fewshot: int) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-m",
        "lm_eval",
        "run",
        "--model",
        "hf",
        "--model_args",
        f"pretrained={model},dtype=bfloat16,trust_remote_code=False",
        "--tasks",
        "mmlu_nutrition",
        "--num_fewshot",
        str(num_fewshot),
        "--batch_size",
        "auto",
        "--device",
        "cuda:0",
        "--output_path",
        str(output_dir),
        "--log_samples",
    ]
    print("Running:", " ".join(command), flush=True)
    subprocess.run(command, check=True)
    candidates = sorted(output_dir.rglob("results*.json"), key=lambda path: path.stat().st_mtime)
    if not candidates:
        raise RuntimeError(f"lm-eval did not write a results JSON under {output_dir}")
    raw = json.loads(candidates[-1].read_text(encoding="utf-8"))
    task = raw.get("results", {}).get("mmlu_nutrition", {})
    normalized = {
        "model": model,
        "task": "mmlu_nutrition",
        "num_fewshot": num_fewshot,
        "accuracy": task.get("acc,none", task.get("acc")),
        "accuracy_stderr": task.get("acc_stderr,none", task.get("acc_stderr")),
        "source_result": str(candidates[-1]),
    }
    return normalized


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the standardized MMLU nutrition benchmark")
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-fewshot", type=int, default=5)
    args = parser.parse_args()
    result = run_mmlu(args.model, args.output_dir, args.num_fewshot)
    atomic_write_json(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
