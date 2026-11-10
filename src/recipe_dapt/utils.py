from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def require_cuda_bf16() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the training and evaluation commands")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("The selected CUDA device does not report BF16 support")


def discover_olm_checkpoint(checkpoint_dir: Path) -> tuple[int, Path] | None:
    checkpoints: list[tuple[int, Path]] = []
    for path in checkpoint_dir.glob("step_*.pt"):
        try:
            step = int(path.stem.split("_", 1)[1])
        except (IndexError, ValueError):
            continue
        checkpoints.append((step, path))
    return max(checkpoints) if checkpoints else None
