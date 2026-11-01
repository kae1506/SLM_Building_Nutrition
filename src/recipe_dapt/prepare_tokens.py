from __future__ import annotations

import argparse
import json

from .config import load_config
from .packing import prepare_recipe_tokens


def main() -> None:
    parser = argparse.ArgumentParser(description="Tokenize and pack the cleaned recipe corpus")
    parser.add_argument("--config", required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    manifest = prepare_recipe_tokens(
        input_root=config.paths["recipe_source"],
        output_root=config.paths["recipe_tokens"],
        model_name=config.model_name,
        sequence_length=int(config.data["sequence_length"]),
        read_batch_size=int(config.data["tokenizer_batch_size"]),
        blocks_per_shard=int(config.data["token_blocks_per_shard"]),
        overwrite=args.overwrite,
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
