from __future__ import annotations

import hashlib
import json
import shutil
import sys
import time
from itertools import chain
from pathlib import Path
from typing import Iterable

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from transformers import AutoTokenizer

from .utils import atomic_write_json


class PackedParquetWriter:
    def __init__(self, root: Path, split: str, sequence_length: int, blocks_per_shard: int) -> None:
        self.root = root / split
        self.root.mkdir(parents=True, exist_ok=True)
        self.sequence_length = sequence_length
        self.blocks_per_shard = blocks_per_shard
        self.pending: list[np.ndarray] = []
        self.pending_blocks = 0
        self.total_blocks = 0
        self.shards = 0

    @property
    def block_count(self) -> int:
        return self.total_blocks + self.pending_blocks

    def add(self, blocks: np.ndarray) -> None:
        if blocks.size == 0:
            return
        cursor = 0
        while cursor < len(blocks):
            capacity = self.blocks_per_shard - self.pending_blocks
            take = min(capacity, len(blocks) - cursor)
            self.pending.append(np.ascontiguousarray(blocks[cursor : cursor + take]))
            self.pending_blocks += take
            cursor += take
            if self.pending_blocks == self.blocks_per_shard:
                self.flush()

    def flush(self) -> None:
        if not self.pending:
            return
        blocks = np.concatenate(self.pending, axis=0)
        flat = pa.array(blocks.reshape(-1), type=pa.uint32())
        packed = pa.FixedSizeListArray.from_arrays(flat, self.sequence_length)
        table = pa.Table.from_arrays([packed], names=["input_ids"])
        path = self.root / f"part-{self.shards:05d}.parquet"
        pq.write_table(
            table,
            path,
            compression="zstd",
            compression_level=3,
            row_group_size=1024,
            write_statistics=True,
        )
        self.total_blocks += len(blocks)
        self.shards += 1
        self.pending.clear()
        self.pending_blocks = 0

    def close(self) -> None:
        self.flush()


class ContinuousPacker:
    def __init__(self, writer: PackedParquetWriter, eos_id: int) -> None:
        self.writer = writer
        self.eos_id = eos_id
        self.remainder = np.empty(0, dtype=np.uint32)
        self.documents = 0
        self.raw_tokens = 0

    def add_encoded(self, encoded: list[list[int]], max_blocks: int | None = None) -> bool:
        for ids in encoded:
            self.documents += 1
            self.raw_tokens += len(ids) + 1
            stream = np.fromiter(chain(ids, (self.eos_id,)), dtype=np.uint32, count=len(ids) + 1)
            if self.remainder.size:
                stream = np.concatenate((self.remainder, stream))
            usable_blocks = stream.size // self.writer.sequence_length
            if max_blocks is not None:
                usable_blocks = min(usable_blocks, max_blocks - self.writer.block_count)
            usable = usable_blocks * self.writer.sequence_length
            if usable:
                self.writer.add(stream[:usable].reshape(-1, self.writer.sequence_length))
            self.remainder = stream[usable:].copy()
            if max_blocks is not None and self.writer.block_count >= max_blocks:
                return True
        return False

    def stats(self, elapsed: float) -> dict[str, int | float]:
        packed_tokens = self.writer.block_count * self.writer.sequence_length
        return {
            "documents": self.documents,
            "tokens_including_eos": self.raw_tokens,
            "packed_blocks": self.writer.block_count,
            "packed_tokens": packed_tokens,
            "dropped_final_tokens": int(self.remainder.size),
            "output_shards": self.writer.shards,
            "elapsed_seconds": round(elapsed, 3),
        }


def load_tokenizer(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if not tokenizer.is_fast:
        raise RuntimeError("A Hugging Face fast tokenizer is required")
    if tokenizer.eos_token_id is None:
        raise RuntimeError("Tokenizer has no EOS token for document separation")
    tokenizer.model_max_length = sys.maxsize
    return tokenizer


def tokenizer_metadata(tokenizer) -> dict[str, object]:
    backend = tokenizer.backend_tokenizer.to_str().encode("utf-8")
    return {
        "tokenizer_class": tokenizer.__class__.__name__,
        "tokenizer_backend_sha256": hashlib.sha256(backend).hexdigest(),
        "vocabulary_size": len(tokenizer),
        "eos_token": tokenizer.eos_token,
        "eos_token_id": tokenizer.eos_token_id,
    }


def prepare_recipe_tokens(
    input_root: Path,
    output_root: Path,
    model_name: str,
    sequence_length: int,
    read_batch_size: int,
    blocks_per_shard: int,
    overwrite: bool = False,
) -> dict:
    manifest_path = output_root / "manifest.json"
    if manifest_path.is_file() and not overwrite:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["tokenizer"] != model_name or int(manifest["sequence_length"]) != sequence_length:
            raise RuntimeError(f"Existing token data is incompatible: {manifest_path}")
        return manifest
    if output_root.exists():
        if not overwrite:
            raise FileExistsError(f"Output exists without a valid manifest: {output_root}")
        _safe_remove_output(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    tokenizer = load_tokenizer(model_name)
    started_all = time.monotonic()
    splits: dict[str, dict] = {}
    for split in ("train", "validation", "test"):
        paths = sorted((input_root / split).glob("*.parquet"))
        if not paths:
            raise FileNotFoundError(f"No source shards found for {split}: {input_root / split}")
        writer = PackedParquetWriter(output_root, split, sequence_length, blocks_per_shard)
        packer = ContinuousPacker(writer, tokenizer.eos_token_id)
        started = time.monotonic()
        for file_number, path in enumerate(paths, start=1):
            parquet = pq.ParquetFile(path)
            for batch in parquet.iter_batches(batch_size=read_batch_size, columns=["text"]):
                texts = batch.column(0).to_pylist()
                encoded = tokenizer(
                    texts,
                    add_special_tokens=False,
                    return_attention_mask=False,
                    return_token_type_ids=False,
                )["input_ids"]
                packer.add_encoded(encoded)
            print(
                f"{split}: {file_number}/{len(paths)} shards, "
                f"{packer.documents:,} docs, {packer.raw_tokens:,} tokens",
                flush=True,
            )
        writer.close()
        splits[split] = packer.stats(time.monotonic() - started)

    manifest = {
        "format_version": 1,
        "description": "EOS-separated packed Recipe1M + RecipeNLG tokens for Qwen DAPT",
        "source_corpus": str(input_root.resolve()),
        "tokenizer": model_name,
        **tokenizer_metadata(tokenizer),
        "sequence_length": sequence_length,
        "stored_dtype": "uint32",
        "stored_columns": ["input_ids"],
        "splits": splits,
        "totals": {
            "documents": sum(int(item["documents"]) for item in splits.values()),
            "packed_blocks": sum(int(item["packed_blocks"]) for item in splits.values()),
            "packed_tokens": sum(int(item["packed_tokens"]) for item in splits.values()),
        },
        "elapsed_seconds": round(time.monotonic() - started_all, 3),
    }
    atomic_write_json(manifest_path, manifest)
    return manifest


def _safe_remove_output(path: Path) -> None:
    resolved = path.resolve()
    if len(resolved.parts) < 5 or resolved == Path(resolved.anchor):
        raise RuntimeError(f"Refusing to remove unsafe output path: {resolved}")
    shutil.rmtree(resolved)
