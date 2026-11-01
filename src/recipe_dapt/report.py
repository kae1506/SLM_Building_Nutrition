from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .config import ExperimentConfig, load_config


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _number(value: Any, digits: int = 4) -> str:
    if value is None:
        return "not run"
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, float):
        return f"{value:,.{digits}f}"
    return str(value)


def _delta(before: Any, after: Any, *, lower_is_better: bool = False) -> str:
    if before is None or after is None:
        return "—"
    difference = float(after) - float(before)
    sign = "+" if difference >= 0 else ""
    marker = "better" if (difference < 0) == lower_is_better else "worse"
    if difference == 0:
        marker = "unchanged"
    return f"{sign}{difference:.4f} ({marker})"


def _metric_pair(
    lines: list[str],
    label: str,
    before: dict[str, Any] | None,
    after: dict[str, Any] | None,
    key: str,
    *,
    lower_is_better: bool = False,
) -> None:
    before_value = None if before is None else before.get(key)
    after_value = None if after is None else after.get(key)
    lines.append(
        f"| {label} | {_number(before_value)} | {_number(after_value)} | "
        f"{_delta(before_value, after_value, lower_is_better=lower_is_better)} |"
    )


def build_report(config: ExperimentConfig) -> str:
    output = config.output_dir
    evaluation = output / "evaluation"
    recipe_manifest = _read_json(config.paths["recipe_tokens"] / "manifest.json")
    replay_manifest = _read_json(config.paths["replay_tokens"] / "manifest.json")
    source_manifest = _read_json(config.paths["recipe_source"] / "manifest.json")
    training = _read_json(output / "training_summary.json")

    base_recipe = _read_json(evaluation / "baseline_recipe.json")
    adapted_recipe = _read_json(evaluation / "adapted_recipe.json")
    base_replay = _read_json(evaluation / "baseline_replay.json")
    adapted_replay = _read_json(evaluation / "adapted_replay.json")
    base_domain = _read_json(evaluation / "baseline_food.json")
    adapted_domain = _read_json(evaluation / "adapted_food.json")
    base_mmlu = _read_json(evaluation / "baseline_mmlu_nutrition.json")
    adapted_mmlu = _read_json(evaluation / "adapted_mmlu_nutrition.json")

    lines = [
        f"# DAPT report: {config.run_name}",
        "",
        f"Base checkpoint: `{config.model_name}`  ",
        f"Adapted checkpoint: `{output / 'final'}`  ",
        f"General-domain replay: {float(config.data['replay_ratio']):.1%}",
        "",
        "## Corpus and run",
        "",
        "| Item | Value |",
        "|---|---:|",
    ]
    if source_manifest:
        for split, rows in source_manifest.get("output_rows", {}).items():
            lines.append(f"| Clean {split} documents | {_number(rows, 0)} |")
    if recipe_manifest:
        for split, details in recipe_manifest.get("splits", {}).items():
            lines.append(f"| Packed recipe {split} tokens | {_number(details.get('packed_tokens'), 0)} |")
    if replay_manifest:
        lines.append(
            "| Prepared replay train tokens | "
            f"{_number(replay_manifest.get('splits', {}).get('train', {}).get('packed_tokens'), 0)} |"
        )
    lines.extend(
        [
            f"| Sequence length | {_number(config.data['sequence_length'], 0)} |",
            f"| Planned training tokens | {_number(config.resolved_token_budget(), 0)} |",
            f"| Planned optimizer steps | {_number(config.max_steps(), 0)} |",
        ]
    )
    if training:
        lines.extend(
            [
                f"| Completed optimizer steps | {_number(training.get('max_steps'), 0)} |",
                f"| Wall time (hours) | {_number(float(training.get('wall_seconds', 0)) / 3600, 2)} |",
                f"| Training throughput (tokens/s) | {_number(training.get('tokens_per_second'), 1)} |",
                f"| Peak allocated GPU memory (GiB) | {_number(training.get('peak_gpu_memory_gib'), 2)} |",
            ]
        )

    lines.extend(
        [
            "",
            "## Language-model metrics",
            "",
            "The recipe split measures in-domain adaptation. The replay split is a small "
            "general-text sentinel for catastrophic forgetting; it is not a comprehensive general benchmark.",
            "",
            "| Metric | Before | After | Change |",
            "|---|---:|---:|---:|",
        ]
    )
    _metric_pair(lines, "Recipe validation perplexity", base_recipe, adapted_recipe, "perplexity", lower_is_better=True)
    _metric_pair(lines, "Recipe next-token accuracy", base_recipe, adapted_recipe, "next_token_accuracy")
    if float(config.data["replay_ratio"]):
        _metric_pair(lines, "Replay validation perplexity", base_replay, adapted_replay, "perplexity", lower_is_better=True)

    lines.extend(
        [
            "",
            "## Food and nutrition benchmarks",
            "",
            "Food tasks are generated only from the held-out recipe test split and scored by "
            "length-normalized option likelihood. MMLU Nutrition is the standardized 5-shot "
            "multiple-choice benchmark.",
            "",
            "| Benchmark | Before | After | Change |",
            "|---|---:|---:|---:|",
        ]
    )
    _metric_pair(lines, "Held-out food mean accuracy", base_domain, adapted_domain, "mean_accuracy")
    if base_domain or adapted_domain:
        before_tasks = {} if base_domain is None else base_domain.get("tasks", {})
        after_tasks = {} if adapted_domain is None else adapted_domain.get("tasks", {})
        for task in sorted(set(before_tasks) | set(after_tasks)):
            _metric_pair(
                lines,
                task.replace("_", " ").title(),
                before_tasks.get(task),
                after_tasks.get(task),
                "accuracy",
            )
    _metric_pair(lines, "MMLU Nutrition accuracy", base_mmlu, adapted_mmlu, "accuracy")

    lines.extend(
        [
            "",
            "## Interpretation guardrails",
            "",
            "- Prefer lower held-out recipe perplexity and higher food/nutrition accuracy.",
            "- Treat a replay-perplexity increase as a forgetting warning, not definitive proof of regression.",
            "- Compare runs only when tokenizer, evaluation block count, seeds, and benchmark prompts are unchanged.",
            "- This is a base language model checkpoint, so likelihood scoring is more appropriate than chat-style generation grading.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the before/after DAPT Markdown report")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    config = load_config(args.config)
    report_path = args.output or (config.output_dir / "report.md")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(build_report(config), encoding="utf-8")
    print(report_path)


if __name__ == "__main__":
    main()
