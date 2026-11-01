from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset, load_dataset
from torch.utils.data import Dataset as TorchDataset


def parquet_files(root: Path, split: str) -> list[str]:
    files = sorted(str(path) for path in (root / split).glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No Parquet shards found under {root / split}")
    return files


def load_packed_dataset(root: Path, split: str, *, seed: int | None = None) -> Dataset:
    dataset = load_dataset(
        "parquet",
        data_files={split: parquet_files(root, split)},
        split=split,
    )
    if seed is not None:
        dataset = dataset.shuffle(seed=seed)
    return dataset


class MixedPackedDataset(TorchDataset):
    """Fixed-length training view with an exact periodic replay fraction.

    The outer Trainer sampler shuffles this index space. Within the index space,
    every Nth item is replay data, making the mix stable across restarts and GPUs.
    """

    def __init__(
        self,
        recipe: Dataset,
        replay: Dataset | None,
        replay_ratio: float,
        total_examples: int,
    ) -> None:
        self.recipe = recipe
        self.replay = replay
        self.replay_ratio = replay_ratio
        self.total_examples = total_examples
        if replay_ratio:
            if replay is None or len(replay) == 0:
                raise ValueError("Replay data is required when replay_ratio is non-zero")
            self.period = round(1.0 / replay_ratio)
            if not math.isclose(1.0 / self.period, replay_ratio, abs_tol=1e-9):
                raise ValueError("Replay ratio must be representable as one item per integer period")
        else:
            self.period = 0

    def __len__(self) -> int:
        return self.total_examples

    def __getitem__(self, index: int) -> dict[str, Any]:
        if self.period and index % self.period == 0:
            replay_index = (index // self.period) % len(self.replay)
            return self.replay[replay_index]
        general_before = 0 if not self.period else index // self.period + 1
        recipe_index = (index - general_before) % len(self.recipe)
        return self.recipe[recipe_index]


class CausalLMCollator:
    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        input_ids = torch.tensor([feature["input_ids"] for feature in features], dtype=torch.long)
        return {
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids),
            "labels": input_ids.clone(),
        }


def evaluation_subset(root: Path, split: str, blocks: int, seed: int) -> Dataset:
    dataset = load_packed_dataset(root, split, seed=seed)
    return dataset.select(range(min(blocks, len(dataset))))
