from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

from datasets import load_dataset

from .config import load_config
from .packing import (
    ContinuousPacker,
    PackedParquetWriter,
    _safe_remove_output,
    load_tokenizer,
    tokenizer_metadata,
)
from .utils import atomic_write_json


def prepare_replay(config, overwrite: bool = False) -> dict:
    output_root = config.paths["replay_tokens"]
    manifest_path = output_root / "manifest.json"
    if manifest_path.is_file() and not overwrite:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        required = int(config.data["replay_prepare_train_tokens"])
        if int(manifest["splits"]["train"]["packed_tokens"]) >= required:
            return manifest
        raise RuntimeError(
            f"Existing replay set is too small ({manifest_path}); rerun with --overwrite"
        )
    if output_root.exists():
        if not overwrite:
            raise FileExistsError(f"Replay output exists without a valid manifest: {output_root}")
        _safe_remove_output(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    sequence_length = int(config.data["sequence_length"])
    blocks_per_shard = int(config.data["token_blocks_per_shard"])
    train_blocks = math.ceil(int(config.data["replay_prepare_train_tokens"]) / sequence_length)
    validation_blocks = math.ceil(
        int(config.data["replay_prepare_validation_tokens"]) / sequence_length
    )
    tokenizer = load_tokenizer(config.model_name)
    stream = load_dataset(
        str(config.data["replay_dataset"]),
        split="train",
        streaming=True,
    ).shuffle(seed=int(config.data["seed"]), buffer_size=10_000)

    writers = {
        "validation": PackedParquetWriter(output_root, "validation", sequence_length, blocks_per_shard),
        "train": PackedParquetWriter(output_root, "train", sequence_length, blocks_per_shard),
    }
    packers = {
        split: ContinuousPacker(writer, tokenizer.eos_token_id)
        for split, writer in writers.items()
    }
    targets = {"validation": validation_blocks, "train": train_blocks}
    current_split = "validation"
    pending_texts: list[str] = []
    started = time.monotonic()

    def process_batch() -> None:
        nonlocal current_split, pending_texts
        if not pending_texts:
            return
        encoded = tokenizer(
            pending_texts,
            add_special_tokens=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )["input_ids"]
        offset = 0
        while offset < len(encoded):
            done = packers[current_split].add_encoded(
                encoded[offset : offset + 1], max_blocks=targets[current_split]
            )
            offset += 1
            if done and current_split == "validation":
                current_split = "train"
            elif done and current_split == "train":
                break
        pending_texts = []

    for item in stream:
        text = str(item.get("text", "")).strip()
        if not text:
            continue
        pending_texts.append(text[:100_000])
        if len(pending_texts) >= 256:
            process_batch()
            print(
                f"replay: validation={writers['validation'].block_count:,} blocks, "
                f"train={writers['train'].block_count:,} blocks",
                flush=True,
            )
            if writers["train"].block_count >= train_blocks:
                break
    process_batch()
    for writer in writers.values():
        writer.close()
    if writers["train"].block_count < train_blocks:
        raise RuntimeError("Replay stream ended before reaching the requested token budget")

    elapsed = time.monotonic() - started
    splits = {split: packer.stats(elapsed) for split, packer in packers.items()}
    manifest = {
        "format_version": 1,
        "description": "FineWeb-Edu general-domain replay packed for Qwen DAPT",
        "source_dataset": config.data["replay_dataset"],
        "tokenizer": config.model_name,
        **tokenizer_metadata(tokenizer),
        "sequence_length": sequence_length,
        "splits": splits,
        "seed": int(config.data["seed"]),
        "elapsed_seconds": round(elapsed, 3),
    }
    atomic_write_json(manifest_path, manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare a deterministic general-domain replay sample")
    parser.add_argument("--config", required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    print(json.dumps(prepare_replay(config, overwrite=args.overwrite), indent=2))


if __name__ == "__main__":
    main()
