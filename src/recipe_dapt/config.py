from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ExperimentConfig:
    path: Path
    raw: dict[str, Any]

    @property
    def project_root(self) -> Path:
        return self.path.parent.parent.resolve()

    @property
    def run_name(self) -> str:
        return str(self.raw["run_name"])

    @property
    def model_name(self) -> str:
        return str(self.raw["model_name"])

    @property
    def paths(self) -> dict[str, Path]:
        return {
            key: self._resolve_path(value)
            for key, value in self.raw["paths"].items()
        }

    @property
    def data(self) -> dict[str, Any]:
        return self.raw["data"]

    @property
    def training(self) -> dict[str, Any]:
        return self.raw["training"]

    @property
    def benchmarks(self) -> dict[str, Any]:
        return self.raw["benchmarks"]

    @property
    def output_dir(self) -> Path:
        return self.paths["output_root"] / self.run_name

    def _resolve_path(self, value: str | Path) -> Path:
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = self.project_root / path
        return path.resolve()

    def recipe_manifest(self) -> dict[str, Any]:
        path = self.paths["recipe_tokens"] / "manifest.json"
        if not path.is_file():
            raise FileNotFoundError(f"Token manifest not found: {path}")
        return json.loads(path.read_text(encoding="utf-8"))

    def resolved_token_budget(self) -> int:
        budget = self.training["token_budget"]
        if budget != "recipe_corpus":
            return int(budget)
        recipe_tokens = int(self.recipe_manifest()["splits"]["train"]["packed_tokens"])
        replay_ratio = float(self.data["replay_ratio"])
        return math.ceil(recipe_tokens / (1.0 - replay_ratio))

    def tokens_per_step(self) -> int:
        return int(self.training["global_batch_sequences"]) * int(self.data["sequence_length"])

    def max_steps(self) -> int:
        return math.ceil(self.resolved_token_budget() / self.tokens_per_step())

    def gradient_accumulation_steps(self, world_size: int | None = None) -> int:
        if world_size is None:
            world_size = int(os.environ.get("WORLD_SIZE", "1"))
        global_batch = int(self.training["global_batch_sequences"])
        micro_batch = int(self.training["per_device_train_batch_size"])
        denominator = micro_batch * world_size
        if global_batch % denominator:
            raise ValueError(
                "global_batch_sequences must be divisible by "
                f"per_device_train_batch_size * world_size ({denominator})"
            )
        return global_batch // denominator

    def resolved_dict(self) -> dict[str, Any]:
        result = json.loads(json.dumps(self.raw))
        result["paths"] = {key: str(value) for key, value in self.paths.items()}
        result["resolved"] = {
            "output_dir": str(self.output_dir),
            "token_budget": self.resolved_token_budget(),
            "tokens_per_step": self.tokens_per_step(),
            "max_steps": self.max_steps(),
            "gradient_accumulation_steps": self.gradient_accumulation_steps(),
        }
        return result


def load_config(path: str | Path) -> ExperimentConfig:
    config_path = Path(path).expanduser().resolve()
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Config must contain a YAML mapping: {config_path}")

    env_overrides = {
        "DAPT_MODEL": ("model_name", str),
        "DAPT_REPLAY_RATIO": ("data.replay_ratio", float),
        "DAPT_ATTENTION": ("training.attention_implementation", str),
        "DAPT_TORCH_COMPILE": ("training.torch_compile", lambda x: x.lower() in {"1", "true", "yes"}),
    }
    for env_name, (dotted_key, cast) in env_overrides.items():
        if env_name not in os.environ:
            continue
        target = raw
        parts = dotted_key.split(".")
        for part in parts[:-1]:
            target = target[part]
        target[parts[-1]] = cast(os.environ[env_name])

    data_root = os.environ.get("DAPT_DATA_ROOT")
    if data_root:
        root = Path(data_root).expanduser().resolve()
        raw["paths"]["recipe_source"] = str(root / "processed" / "dapt-parquet")
        raw["paths"]["recipe_tokens"] = str(root / "tokenized" / "qwen25-0.5b-seq1024")
        raw["paths"]["replay_tokens"] = str(root / "tokenized" / "fineweb-edu-qwen25-seq1024")
    output_root = os.environ.get("DAPT_OUTPUT_ROOT")
    if output_root:
        raw["paths"]["output_root"] = str(Path(output_root).expanduser().resolve())

    config = ExperimentConfig(config_path, raw)
    _validate(config)
    return config


def _validate(config: ExperimentConfig) -> None:
    required_top = {"run_name", "model_name", "paths", "data", "training", "benchmarks"}
    missing = required_top.difference(config.raw)
    if missing:
        raise ValueError(f"Missing config keys: {sorted(missing)}")
    sequence_length = int(config.data["sequence_length"])
    if sequence_length <= 1:
        raise ValueError("sequence_length must be greater than one")
    replay_ratio = float(config.data["replay_ratio"])
    if not 0.0 <= replay_ratio < 0.5:
        raise ValueError("replay_ratio must be in [0, 0.5)")
    if replay_ratio:
        inverse = 1.0 / replay_ratio
        if not math.isclose(inverse, round(inverse), rel_tol=0, abs_tol=1e-8):
            raise ValueError("replay_ratio must be an exact reciprocal such as 0.05 or 0.10")
    if int(config.training["global_batch_sequences"]) <= 0:
        raise ValueError("global_batch_sequences must be positive")
