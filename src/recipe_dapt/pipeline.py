from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

from .config import load_config
from .utils import atomic_write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare, evaluate, train, and report a DAPT run")
    parser.add_argument("--config", required=True)
    parser.add_argument("--skip-preparation", action="store_true")
    parser.add_argument("--skip-baseline", action="store_true")
    parser.add_argument("--skip-mmlu", action="store_true")
    parser.add_argument("--force", action="store_true", help="Rerun completed evaluation and report stages")
    parser.add_argument("--dry-run", action="store_true", help="Print the stage commands without executing them")
    args = parser.parse_args()

    config_path = Path(args.config).expanduser().resolve()
    config = load_config(config_path)
    output = config.output_dir
    evaluation = output / "evaluation"
    output.mkdir(parents=True, exist_ok=True)
    evaluation.mkdir(parents=True, exist_ok=True)

    def command(module: str, *arguments: object) -> list[str]:
        return [sys.executable, "-m", module, *map(str, arguments)]

    def run_stage(
        name: str,
        argv: list[str],
        product: Path | None = None,
        *,
        allow_force: bool = True,
    ) -> None:
        force_stage = args.force and allow_force
        if product is not None and product.exists() and not force_stage:
            print(f"[{name}] already complete: {product}", flush=True)
            return
        rendered = shlex.join(argv)
        print(f"[{name}] {rendered}", flush=True)
        if not args.dry_run:
            subprocess.run(argv, check=True, env=os.environ.copy())

    if not args.skip_preparation:
        run_stage(
            "download and clean text corpus",
            command("recipe_dapt.bootstrap_data", "--config", config_path),
            config.paths["recipe_source"] / "manifest.json",
        )
        run_stage(
            "prepare recipe tokens",
            command("recipe_dapt.prepare_tokens", "--config", config_path),
            config.paths["recipe_tokens"] / "manifest.json",
        )
        if float(config.data["replay_ratio"]):
            run_stage(
                "prepare replay tokens",
                command("recipe_dapt.prepare_replay", "--config", config_path),
                config.paths["replay_tokens"] / "manifest.json",
            )

    if not args.dry_run:
        resolved = config.resolved_dict()
        atomic_write_json(output / "run_plan.json", resolved)
        print(
            f"[{config.run_name}] {resolved['resolved']['token_budget']:,} tokens, "
            f"{resolved['resolved']['max_steps']:,} optimizer steps",
            flush=True,
        )
    else:
        try:
            print(json.dumps(config.resolved_dict()["resolved"], indent=2))
        except FileNotFoundError:
            print("Resolved corpus-pass steps will be calculated after recipe tokenization.")

    base_model = config.model_name
    final_model = output / "final"
    mmlu_enabled = bool(config.benchmarks.get("run_mmlu_nutrition", True)) and not args.skip_mmlu
    fewshot = int(config.benchmarks.get("mmlu_nutrition_num_fewshot", 5))

    def evaluations(prefix: str, model: str | Path) -> None:
        run_stage(
            f"{prefix} recipe perplexity",
            command(
                "recipe_dapt.evaluate", "--config", config_path, "--model", model,
                "--dataset", "recipe", "--output", evaluation / f"{prefix}_recipe.json",
            ),
            evaluation / f"{prefix}_recipe.json",
        )
        if float(config.data["replay_ratio"]):
            run_stage(
                f"{prefix} replay perplexity",
                command(
                    "recipe_dapt.evaluate", "--config", config_path, "--model", model,
                    "--dataset", "replay", "--output", evaluation / f"{prefix}_replay.json",
                ),
                evaluation / f"{prefix}_replay.json",
            )
        run_stage(
            f"{prefix} held-out food benchmark",
            command(
                "recipe_dapt.domain_benchmark", "--config", config_path, "--model", model,
                "--output", evaluation / f"{prefix}_food.json",
            ),
            evaluation / f"{prefix}_food.json",
        )
        if mmlu_enabled:
            run_stage(
                f"{prefix} MMLU nutrition",
                command(
                    "recipe_dapt.mmlu", "--model", model,
                    "--output-dir", evaluation / f"{prefix}_mmlu_raw",
                    "--output", evaluation / f"{prefix}_mmlu_nutrition.json",
                    "--num-fewshot", fewshot,
                ),
                evaluation / f"{prefix}_mmlu_nutrition.json",
            )

    if not args.skip_baseline:
        evaluations("baseline", base_model)

    train_command = command("recipe_dapt.train", "--config", config_path, "--resume")
    run_stage("DAPT training", train_command, final_model, allow_force=False)
    evaluations("adapted", final_model)
    run_stage(
        "write report",
        command("recipe_dapt.report", "--config", config_path, "--output", output / "report.md"),
        output / "report.md",
    )
    if not args.dry_run:
        print(f"Complete. Report: {output / 'report.md'}", flush=True)


if __name__ == "__main__":
    main()
