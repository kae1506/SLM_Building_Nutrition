from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import urllib.request
from pathlib import Path

from .config import load_config
from .utils import atomic_write_json


RECIPE1M_URL = "http://wednesday.csail.mit.edu/temporal/release/recipe1M_layers.tar.gz"
RECIPE1M_ARCHIVE_SHA256 = "e180260dcd438be96e63c1c68b5f09a6415d1d1c51b1f2c2bc0b08079cb5d6c3"
RECIPE1M_JSON_SHA256 = "ad76f06eb4e191194fe2c20506bbd5ac9a4f1be5ec0241916b47be0433f02a98"
RECIPENLG_HANDLE = "paultimothymooney/recipenlg/versions/1"
RECIPENLG_CSV_SHA256 = "f376ee0bb9f99f323761abf6270d453f4eff2002c3625d04cabcf8b57dca0803"


def sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def verify(path: Path, expected: str) -> None:
    actual = sha256(path)
    if actual != expected:
        raise RuntimeError(f"Checksum mismatch for {path}: expected {expected}, got {actual}")


def download_resumable(url: str, target: Path, expected_sha256: str) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".partial")
    if target.is_file():
        verify(target, expected_sha256)
        print(f"Already downloaded: {target}", flush=True)
        return

    offset = partial.stat().st_size if partial.exists() else 0
    headers = {"User-Agent": "recipe-dapt/0.1"}
    if offset:
        headers["Range"] = f"bytes={offset}-"
    request = urllib.request.Request(url, headers=headers)
    try:
        response = urllib.request.urlopen(request, timeout=60)
    except Exception as exc:
        raise RuntimeError(f"Download failed: {url}") from exc

    status = getattr(response, "status", response.getcode())
    mode = "ab" if offset and status == 206 else "wb"
    if mode == "wb":
        offset = 0
    total_header = response.headers.get("Content-Length")
    total = offset + int(total_header) if total_header else None
    written = offset
    with response, partial.open(mode) as destination:
        while True:
            chunk = response.read(8 * 1024 * 1024)
            if not chunk:
                break
            destination.write(chunk)
            written += len(chunk)
            if total:
                print(
                    f"Downloading {target.name}: {written / 1024**3:.2f}/"
                    f"{total / 1024**3:.2f} GiB",
                    flush=True,
                )
            else:
                print(f"Downloading {target.name}: {written / 1024**3:.2f} GiB", flush=True)
    verify(partial, expected_sha256)
    os.replace(partial, target)


def extract_recipe1m(archive: Path, target: Path) -> None:
    if target.is_file():
        verify(target, RECIPE1M_JSON_SHA256)
        print(f"Already extracted: {target}", flush=True)
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".json.partial")
    with tarfile.open(archive, "r:gz") as bundle:
        members = [member for member in bundle.getmembers() if Path(member.name).name == "layer1.json"]
        if len(members) != 1 or not members[0].isfile():
            raise RuntimeError("Recipe1M archive does not contain exactly one layer1.json file")
        source = bundle.extractfile(members[0])
        if source is None:
            raise RuntimeError("Could not read layer1.json from Recipe1M archive")
        with source, temporary.open("wb") as destination:
            shutil.copyfileobj(source, destination, length=8 * 1024 * 1024)
    verify(temporary, RECIPE1M_JSON_SHA256)
    os.replace(temporary, target)


def download_recipenlg(target: Path) -> None:
    if target.is_file():
        verify(target, RECIPENLG_CSV_SHA256)
        print(f"Already downloaded: {target}", flush=True)
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        import kagglehub

        downloaded = Path(
            kagglehub.dataset_download(
                RECIPENLG_HANDLE,
                path="RecipeNLG_dataset.csv",
                output_dir=str(target.parent),
            )
        )
    except Exception as exc:
        raise RuntimeError(
            "RecipeNLG download failed. Accept any Kaggle dataset terms and set "
            "KAGGLE_API_TOKEN, or pass --recipenlg /path/to/RecipeNLG_dataset.csv."
        ) from exc
    candidates = [downloaded] if downloaded.is_file() else list(downloaded.rglob("RecipeNLG_dataset.csv"))
    if target.is_file():
        candidates.insert(0, target)
    if not candidates:
        candidates = list(target.parent.rglob("RecipeNLG_dataset.csv"))
    if not candidates:
        raise RuntimeError(f"Kaggle download completed but RecipeNLG_dataset.csv was not found under {target.parent}")
    source = candidates[0].resolve()
    if source != target.resolve():
        temporary = target.with_suffix(".csv.partial")
        shutil.copyfile(source, temporary)
        os.replace(temporary, target)
    verify(target, RECIPENLG_CSV_SHA256)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and clean the text-only recipe corpus")
    parser.add_argument("--config", required=True)
    parser.add_argument("--recipe1m", type=Path, help="Use a pre-staged layer1.json")
    parser.add_argument("--recipenlg", type=Path, help="Use a pre-staged RecipeNLG_dataset.csv")
    parser.add_argument("--skip-download", action="store_true", help="Require pre-staged raw files")
    parser.add_argument("--overwrite-clean", action="store_true")
    parser.add_argument("--limit-per-source", type=int, help="Only for a data-pipeline smoke test")
    args = parser.parse_args()

    config = load_config(args.config)
    corpus_root = config.paths["recipe_source"].parents[1]
    raw_root = corpus_root / "raw"
    recipe1m_json = args.recipe1m.expanduser().resolve() if args.recipe1m else raw_root / "recipe1m" / "layer1.json"
    recipenlg_csv = args.recipenlg.expanduser().resolve() if args.recipenlg else raw_root / "recipenlg" / "RecipeNLG_dataset.csv"
    archive = raw_root / "recipe1m" / "recipe1M_layers.tar.gz"
    manifest = config.paths["recipe_source"] / "manifest.json"

    if manifest.is_file() and not args.overwrite_clean:
        print(f"Clean corpus already exists: {manifest}")
        return

    if not args.recipe1m and not recipe1m_json.is_file():
        if args.skip_download:
            raise FileNotFoundError(f"Pre-staged Recipe1M file not found: {recipe1m_json}")
        download_resumable(RECIPE1M_URL, archive, RECIPE1M_ARCHIVE_SHA256)
        extract_recipe1m(archive, recipe1m_json)
    if not args.recipenlg and not recipenlg_csv.is_file():
        if args.skip_download:
            raise FileNotFoundError(f"Pre-staged RecipeNLG file not found: {recipenlg_csv}")
        download_recipenlg(recipenlg_csv)
    for path in (recipe1m_json, recipenlg_csv):
        if not path.is_file():
            raise FileNotFoundError(path)
    if not args.recipe1m:
        verify(recipe1m_json, RECIPE1M_JSON_SHA256)
    if not args.recipenlg:
        verify(recipenlg_csv, RECIPENLG_CSV_SHA256)

    command = [
        sys.executable,
        "-m",
        "recipe_dapt.clean_corpus",
        "--recipe1m",
        str(recipe1m_json),
        "--recipenlg",
        str(recipenlg_csv),
        "--output",
        str(config.paths["recipe_source"]),
    ]
    if args.overwrite_clean:
        command.append("--overwrite")
    if args.limit_per_source is not None:
        command.extend(["--limit-per-source", str(args.limit_per_source)])
    print("Cleaning corpus:", " ".join(command), flush=True)
    subprocess.run(command, check=True)

    source_record = {
        "recipe1m": {
            "source_url": RECIPE1M_URL,
            "text_file": str(recipe1m_json),
            "text_sha256": sha256(recipe1m_json),
        },
        "recipenlg": {
            "kaggle_handle": RECIPENLG_HANDLE,
            "text_file": str(recipenlg_csv),
            "text_sha256": sha256(recipenlg_csv),
        },
    }
    atomic_write_json(config.paths["recipe_source"] / "source_manifest.json", source_record)
    print(json.dumps(source_record, indent=2))


if __name__ == "__main__":
    main()
