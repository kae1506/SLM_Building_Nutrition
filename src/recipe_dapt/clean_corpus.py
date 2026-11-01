#!/usr/bin/env python3
"""Prepare Recipe1M and RecipeNLG text as clean, sharded Parquet for DAPT.

Recipe1M is processed first so an exact RecipeNLG copy of an official
Recipe1M validation/test record cannot enter the training split.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import html
import json
import math
import re
import shutil
import sqlite3
import sys
import time
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Iterator

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

try:
    import ijson
except ImportError as exc:  # pragma: no cover - exercised by environment setup
    raise SystemExit("Missing dependency: install ijson before running this script") from exc


CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
HTML_TAG_RE = re.compile(r"<[^>]{1,500}>")
INLINE_SPACE_RE = re.compile(r"[ \t\f\v]+")
NEWLINES_RE = re.compile(r"\n{3,}")
WORD_RE = re.compile(r"\b\w+\b", re.UNICODE)

SCHEMA = pa.schema(
    [
        ("document_id", pa.string()),
        ("source_dataset", pa.string()),
        ("source_partition", pa.string()),
        ("title", pa.string()),
        ("ingredients", pa.list_(pa.string())),
        ("instructions", pa.list_(pa.string())),
        ("normalized_ingredients", pa.list_(pa.string())),
        ("text", pa.string()),
        ("content_hash", pa.string()),
        ("url_hash", pa.string()),
        ("character_count", pa.int32()),
        ("word_count", pa.int32()),
    ]
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe1m", type=Path, required=True, help="Recipe1M layer1.json")
    parser.add_argument("--recipenlg", type=Path, required=True, help="RecipeNLG CSV")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--recipenlg-source",
        default="Gathered",
        help="RecipeNLG source value to retain; use 'all' for every source",
    )
    parser.add_argument("--csv-chunk-rows", type=int, default=25_000)
    parser.add_argument("--shard-rows", type=int, default=50_000)
    parser.add_argument("--min-ingredients", type=int, default=2)
    parser.add_argument("--min-instructions", type=int, default=1)
    parser.add_argument("--max-characters", type=int, default=50_000)
    parser.add_argument(
        "--limit-per-source",
        type=int,
        default=None,
        help="Optional smoke-test limit; omit for the complete corpus",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def clean_scalar(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    text = html.unescape(str(value))
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = CONTROL_RE.sub("", text)
    text = HTML_TAG_RE.sub(" ", text)
    text = "\n".join(INLINE_SPACE_RE.sub(" ", line).strip() for line in text.split("\n"))
    return NEWLINES_RE.sub("\n\n", text).strip()


def clean_lines(values: Iterable[Any]) -> list[str]:
    cleaned: list[str] = []
    previous = None
    for value in values:
        if isinstance(value, dict):
            value = value.get("text", "")
        line = clean_scalar(value)
        if not line or line == previous:
            continue
        cleaned.append(line)
        previous = line
    return cleaned


def parse_list_cell(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return []
    text = str(value).strip()
    if not text:
        return []
    try:
        parsed = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        try:
            parsed = ast.literal_eval(text)
        except (ValueError, SyntaxError):
            return []
    return parsed if isinstance(parsed, list) else []


def canonical(value: str) -> str:
    value = clean_scalar(value).casefold()
    return " ".join(value.split())


def make_document(title: str, ingredients: list[str], instructions: list[str]) -> str:
    ingredient_block = "\n".join(f"- {item}" for item in ingredients)
    instruction_block = "\n".join(
        f"{number}. {step}" for number, step in enumerate(instructions, start=1)
    )
    return (
        f"Recipe: {title}\n\n"
        f"Ingredients:\n{ingredient_block}\n\n"
        f"Instructions:\n{instruction_block}"
    )


def content_digest(title: str, ingredients: list[str], instructions: list[str]) -> bytes:
    payload = json.dumps(
        {
            "title": canonical(title),
            "ingredients": sorted(canonical(item) for item in ingredients),
            "instructions": [canonical(step) for step in instructions],
        },
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.blake2b(payload, digest_size=20).digest()


def url_digest(url: str) -> str:
    normalized = canonical(url).rstrip("/")
    if not normalized:
        return ""
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def recipenlg_split(digest: bytes) -> str:
    bucket = int.from_bytes(digest[:8], "big") % 1000
    if bucket < 10:
        return "validation"
    if bucket < 20:
        return "test"
    return "train"


def recipe1m_split(partition: str) -> str:
    value = canonical(partition)
    return {"train": "train", "val": "validation", "test": "test"}.get(value, "train")


class ExactDeduper:
    def __init__(self, path: Path) -> None:
        self.connection = sqlite3.connect(path)
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA synchronous=NORMAL")
        self.connection.execute("PRAGMA temp_store=MEMORY")
        self.connection.execute("CREATE TABLE IF NOT EXISTS seen (digest BLOB PRIMARY KEY) WITHOUT ROWID")

    def add(self, digest: bytes) -> bool:
        cursor = self.connection.execute("INSERT OR IGNORE INTO seen(digest) VALUES (?)", (digest,))
        return cursor.rowcount == 1

    def commit(self) -> None:
        self.connection.commit()

    def close(self) -> None:
        self.connection.commit()
        self.connection.close()


class ParquetSharder:
    def __init__(self, root: Path, rows_per_shard: int) -> None:
        self.root = root
        self.rows_per_shard = rows_per_shard
        self.buffers: dict[str, list[dict[str, Any]]] = {
            "train": [],
            "validation": [],
            "test": [],
        }
        self.shard_numbers = Counter()
        self.row_counts = Counter()

    def add(self, split: str, record: dict[str, Any]) -> None:
        self.buffers[split].append(record)
        if len(self.buffers[split]) >= self.rows_per_shard:
            self.flush(split)

    def flush(self, split: str) -> None:
        records = self.buffers[split]
        if not records:
            return
        split_dir = self.root / split
        split_dir.mkdir(parents=True, exist_ok=True)
        shard_number = self.shard_numbers[split]
        output = split_dir / f"part-{shard_number:05d}.parquet"
        table = pa.Table.from_pylist(records, schema=SCHEMA)
        pq.write_table(
            table,
            output,
            compression="zstd",
            compression_level=6,
            use_dictionary=["source_dataset", "source_partition"],
            write_statistics=True,
        )
        self.row_counts[split] += len(records)
        self.shard_numbers[split] += 1
        records.clear()

    def close(self) -> None:
        for split in self.buffers:
            self.flush(split)


def validated_record(
    *,
    document_id: str,
    source_dataset: str,
    source_partition: str,
    title: Any,
    ingredients: Iterable[Any],
    instructions: Iterable[Any],
    normalized_ingredients: Iterable[Any],
    url: Any,
    min_ingredients: int,
    min_instructions: int,
    max_characters: int,
) -> tuple[dict[str, Any] | None, str | None, bytes | None]:
    clean_title = clean_scalar(title)
    clean_ingredients = clean_lines(ingredients)
    clean_instructions = clean_lines(instructions)
    clean_normalized = clean_lines(normalized_ingredients)

    if not clean_title:
        return None, "missing_title", None
    if len(clean_ingredients) < min_ingredients:
        return None, "too_few_ingredients", None
    if len(clean_instructions) < min_instructions:
        return None, "too_few_instructions", None

    text = make_document(clean_title, clean_ingredients, clean_instructions)
    if len(text) > max_characters:
        return None, "too_long", None

    digest = content_digest(clean_title, clean_ingredients, clean_instructions)
    record = {
        "document_id": document_id,
        "source_dataset": source_dataset,
        "source_partition": source_partition,
        "title": clean_title,
        "ingredients": clean_ingredients,
        "instructions": clean_instructions,
        "normalized_ingredients": clean_normalized,
        "text": text,
        "content_hash": digest.hex(),
        "url_hash": url_digest(clean_scalar(url)),
        "character_count": len(text),
        "word_count": len(WORD_RE.findall(text)),
    }
    return record, None, digest


def iter_recipe1m(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("rb") as source:
        yield from ijson.items(source, "item")


def process_recipe1m(
    args: argparse.Namespace,
    deduper: ExactDeduper,
    sharder: ParquetSharder,
    stats: Counter,
) -> None:
    print(f"Processing Recipe1M: {args.recipe1m}", flush=True)
    started = time.monotonic()
    for index, raw in enumerate(iter_recipe1m(args.recipe1m), start=1):
        if args.limit_per_source is not None and stats["recipe1m.input"] >= args.limit_per_source:
            break
        stats["recipe1m.input"] += 1
        record, rejection, digest = validated_record(
            document_id=f"recipe1m:{raw.get('id', index)}",
            source_dataset="recipe1m",
            source_partition=clean_scalar(raw.get("partition")),
            title=raw.get("title"),
            ingredients=raw.get("ingredients") or [],
            instructions=raw.get("instructions") or [],
            normalized_ingredients=[],
            url=raw.get("url"),
            min_ingredients=args.min_ingredients,
            min_instructions=args.min_instructions,
            max_characters=args.max_characters,
        )
        if rejection:
            stats[f"recipe1m.rejected.{rejection}"] += 1
            continue
        assert record is not None and digest is not None
        if not deduper.add(digest):
            stats["recipe1m.rejected.exact_duplicate"] += 1
            continue
        split = recipe1m_split(record["source_partition"])
        sharder.add(split, record)
        stats[f"recipe1m.output.{split}"] += 1
        if index % 100_000 == 0:
            deduper.commit()
            elapsed = time.monotonic() - started
            print(f"  Recipe1M: {index:,} input rows in {elapsed / 60:.1f} min", flush=True)
    deduper.commit()


def process_recipenlg(
    args: argparse.Namespace,
    deduper: ExactDeduper,
    sharder: ParquetSharder,
    stats: Counter,
) -> None:
    print(f"Processing RecipeNLG: {args.recipenlg}", flush=True)
    started = time.monotonic()
    chunks = pd.read_csv(
        args.recipenlg,
        chunksize=args.csv_chunk_rows,
        low_memory=False,
        on_bad_lines="skip",
    )
    for chunk_number, chunk in enumerate(chunks, start=1):
        required = {"title", "ingredients", "directions"}
        missing = required.difference(chunk.columns)
        if missing:
            raise ValueError(f"RecipeNLG CSV is missing columns: {sorted(missing)}")

        for raw in chunk.to_dict(orient="records"):
            if args.limit_per_source is not None and stats["recipenlg.input"] >= args.limit_per_source:
                deduper.commit()
                return
            stats["recipenlg.input"] += 1
            row_number = stats["recipenlg.input"]
            source = clean_scalar(raw.get("source"))
            if args.recipenlg_source.casefold() != "all" and source.casefold() != args.recipenlg_source.casefold():
                stats["recipenlg.rejected.source_filter"] += 1
                continue

            external_id = None
            for column in chunk.columns:
                if str(column).startswith("Unnamed:"):
                    external_id = raw.get(column)
                    break
            if external_id is None or (isinstance(external_id, float) and math.isnan(external_id)):
                external_id = row_number
            elif isinstance(external_id, float) and external_id.is_integer():
                external_id = int(external_id)

            record, rejection, digest = validated_record(
                document_id=f"recipenlg:{external_id}",
                source_dataset="recipenlg",
                source_partition=source,
                title=raw.get("title"),
                ingredients=parse_list_cell(raw.get("ingredients")),
                instructions=parse_list_cell(raw.get("directions")),
                normalized_ingredients=parse_list_cell(raw.get("NER")),
                url=raw.get("link"),
                min_ingredients=args.min_ingredients,
                min_instructions=args.min_instructions,
                max_characters=args.max_characters,
            )
            if rejection:
                stats[f"recipenlg.rejected.{rejection}"] += 1
                continue
            assert record is not None and digest is not None
            if not deduper.add(digest):
                stats["recipenlg.rejected.exact_duplicate"] += 1
                continue
            split = recipenlg_split(digest)
            sharder.add(split, record)
            stats[f"recipenlg.output.{split}"] += 1

        deduper.commit()
        elapsed = time.monotonic() - started
        print(
            f"  RecipeNLG: {stats['recipenlg.input']:,} input rows "
            f"({chunk_number} chunks) in {elapsed / 60:.1f} min",
            flush=True,
        )


def main() -> int:
    args = parse_args()
    for input_path in (args.recipe1m, args.recipenlg):
        if not input_path.is_file():
            raise SystemExit(f"Input not found: {input_path}")

    if args.output.exists():
        if not args.overwrite:
            raise SystemExit(f"Output exists; pass --overwrite to replace it: {args.output}")
        shutil.rmtree(args.output)
    args.output.mkdir(parents=True)

    started = time.monotonic()
    stats: Counter = Counter()
    deduper = ExactDeduper(args.output / "dedup.sqlite3")
    sharder = ParquetSharder(args.output, args.shard_rows)

    try:
        process_recipe1m(args, deduper, sharder, stats)
        process_recipenlg(args, deduper, sharder, stats)
        sharder.close()
    finally:
        deduper.close()

    elapsed = time.monotonic() - started
    manifest = {
        "format_version": 1,
        "description": "Clean Recipe1M + RecipeNLG text corpus for domain-adaptive pretraining",
        "recipe1m_input": str(args.recipe1m.resolve()),
        "recipenlg_input": str(args.recipenlg.resolve()),
        "recipenlg_source_filter": args.recipenlg_source,
        "limit_per_source": args.limit_per_source,
        "split_policy": {
            "recipe1m": "official partition (val renamed validation)",
            "recipenlg": "deterministic content hash: 98% train, 1% validation, 1% test",
        },
        "deduplication": "exact canonical title + sorted ingredients + ordered instructions",
        "filters": {
            "min_ingredients": args.min_ingredients,
            "min_instructions": args.min_instructions,
            "max_characters": args.max_characters,
        },
        "schema": str(SCHEMA),
        "parquet_compression": "zstd level 6",
        "elapsed_seconds": round(elapsed, 3),
        "statistics": dict(sorted(stats.items())),
        "output_rows": dict(sorted(sharder.row_counts.items())),
        "output_shards": dict(sorted(sharder.shard_numbers.items())),
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    # The dedup database is an intermediate implementation detail and can be large.
    for suffix in ("", "-shm", "-wal"):
        candidate = args.output / f"dedup.sqlite3{suffix}"
        if candidate.exists():
            candidate.unlink()

    print(json.dumps(manifest, indent=2, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
