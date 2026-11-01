# Qwen2.5 recipe DAPT

This is a clone-and-run domain-adaptive pretraining project for `Qwen/Qwen2.5-0.5B`. From a fresh Git checkout it downloads the text portions of Recipe1M and RecipeNLG, cleans and deduplicates them into Parquet, tokenizes with the Qwen tokenizer, packs fixed 1,024-token examples, mixes in 5% FineWeb-Edu replay, performs full-parameter causal-LM training, and writes a before/after evaluation report. Dataset files, caches, checkpoints, and environments are ignored by Git.

The training data remains causal language-model text—not instruction examples. Every source document is separated by EOS before continuous packing; padding tokens are never used during training.

## Fresh remote machine: start the short pilot

Clone the repository on the CUDA machine and run the pilot from this directory:

```bash
./run_short_pilot.sh
```

That command creates `.venv` when necessary, installs the project, obtains and cleans the text datasets, prepares replay and packed tokens, runs baseline evaluation, performs DAPT, reruns evaluation, and generates the report. It automatically resumes the newest Trainer checkpoint after interruption and skips completed stages. Outputs go to `outputs/short_pilot/`; the final summary is `outputs/short_pilot/report.md`.

RecipeNLG is obtained through KaggleHub. Public downloads may work anonymously; if Kaggle asks for authentication or dataset consent, create an API token in Kaggle settings and export it before launching:

```bash
export KAGGLE_API_TOKEN='your-token'
./run_short_pilot.sh
```

For a production remote machine, keep large artifacts on local NVMe or scratch rather than the Git checkout:

```bash
export DAPT_DATA_ROOT=/scratch/$USER/recipe-dapt-data
export DAPT_OUTPUT_ROOT=/scratch/$USER/recipe-dapt-runs
export DAPT_CACHE_ROOT=/scratch/$USER/recipe-dapt-cache
./run_short_pilot.sh
```

These variables are optional. Without them, data and outputs live under `data/recipes/` and `outputs/` inside the clone.

The automatically invoked `setup.sh` uses `python3` by default. To select another interpreter, run setup explicitly first:

```bash
PYTHON_BIN=/path/to/python3.11 ./setup.sh
```

The setup requires Python 3.10+ and a recent NVIDIA PyTorch build. If the generic pip resolver chooses the wrong CUDA build for the machine, install the correct PyTorch wheel in `.venv` first, then rerun `.venv/bin/pip install -r requirements.txt` and `.venv/bin/pip install -e .`.

## Run profiles

| Profile | Domain + replay tokens | Optimizer steps | Purpose | Command |
|---|---:|---:|---|---|
| Short pilot | 50M | 191 | Verify loss, throughput, memory, and metric direction | `./run_short_pilot.sh` |
| Recommended first run | 300M | 1,145 | First decision-quality DAPT checkpoint | `./run_first_run.sh` |
| Corpus pass | one expected recipe pass + 5% replay | calculated after tokenization | Consume each packed recipe token once in expectation | `./run_corpus_pass.sh` |

All profiles use a global batch of 256 × 1,024 = 262,144 tokens per optimizer step, BF16, fused AdamW, cosine decay, 2% warmup, a 2e-5 peak learning rate, and 0.1 weight decay. On one GPU, batch size 32 with eight accumulation steps produces that global batch. The corpus run computes its exact step count from the Qwen token manifest; it does not rely on a word-count estimate.

Use the pilot’s measured `tokens_per_second` to forecast the longer runs:

```text
training hours = planned tokens / measured pilot tokens_per_second / 3600
```

The full pipeline also includes tokenization and evaluation time, so its wall time is longer than this training-only estimate.

## What gets evaluated

Each profile evaluates both the untouched base checkpoint and the adapted checkpoint:

- Held-out recipe validation loss, perplexity, next-token accuracy, throughput, and peak allocated GPU memory.
- Held-out food knowledge: ingredient-cloze and next-recipe-step selection. Distractors come from other test recipes and choices are scored by length-normalized log likelihood.
- MMLU Nutrition, 5-shot, through EleutherAI's LM Evaluation Harness.
- FineWeb-Edu validation perplexity as a narrow catastrophic-forgetting sentinel when replay is enabled.
- Training wall time, token throughput, peak GPU allocation, best validation checkpoint, token counts, and corpus statistics.

The generated Markdown report records before/after values and deltas. Recipe and replay perplexity are comparable only within this project because tokenization, seed, packed blocks, and prompts are fixed here. The custom food tests are useful regression tests, not established external leaderboards.

Qwen's exact original pretraining mixture is not available here, so “replay” means a deterministic FineWeb-Edu sample, not a reconstruction of Qwen's private/original corpus.

## Data and output flow

```text
Recipe1M layer1.json + RecipeNLG CSV (text only)
  -> clean, validate, exact-deduplicate, deterministic splits
  -> data/recipes/processed/dapt-parquet
  -> Qwen tokenization + EOS + 1,024-token packing
  -> data/recipes/tokenized/qwen25-0.5b-seq1024

HuggingFaceFW/fineweb-edu (streamed sample)
  -> Qwen tokenization + packing
  -> data/recipes/tokenized/fineweb-edu-qwen25-seq1024

base model + 95% recipes + 5% replay
  -> checkpoints + final model
  -> perplexity / food likelihood / MMLU Nutrition
  -> outputs/<profile>/report.md
```

Token Parquet shards contain one fixed-size `input_ids: uint32[1024]` column. Their manifests record exact document, token, block, shard, tokenizer, and tokenizer-backend hash statistics. Raw text does not get copied into the token shards.

The Recipe1M downloader fetches the layers archive but securely extracts only `layer1.json`; it does not download the image archives. The RecipeNLG downloader requests only `RecipeNLG_dataset.csv`. Known SHA-256 checksums are verified before cleaning.

### Pre-staged or offline datasets

If the remote machine cannot access the source sites, stage `layer1.json` and `RecipeNLG_dataset.csv` yourself, then run:

```bash
./setup.sh
.venv/bin/python -m recipe_dapt.bootstrap_data \
  --config configs/short_pilot.yaml \
  --recipe1m /mounted/data/layer1.json \
  --recipenlg /mounted/data/RecipeNLG_dataset.csv
./run_short_pilot.sh
```

After the clean corpus and token manifests exist, all subsequent runs reuse them. A shared `DAPT_DATA_ROOT` can therefore be used by the pilot, first run, and corpus pass.

## Useful controls

```bash
# Preview every pipeline stage without downloads or GPU work
./run_short_pilot.sh --dry-run

# Skip baseline metrics if they were already obtained elsewhere
./run_short_pilot.sh --skip-baseline

# Skip the standardized nutrition benchmark
./run_short_pilot.sh --skip-mmlu

# Disable general replay for an ablation
DAPT_REPLAY_RATIO=0 ./run_short_pilot.sh

# Use FlashAttention 2 after installing flash-attn
DAPT_ATTENTION=flash_attention_2 ./run_short_pilot.sh

# Enable torch.compile after validating the ordinary pilot
DAPT_TORCH_COMPILE=1 ./run_short_pilot.sh

# Force rerunning evaluation and report outputs; trained/token data is preserved
./run_short_pilot.sh --force
```

The Qwen model and FineWeb-Edu replay sample are downloaded from Hugging Face on first use. Recipe1M comes from the MIT release server and RecipeNLG through KaggleHub. No images are downloaded or included.

For logs during training:

```bash
.venv/bin/tensorboard --logdir outputs
```

For a quick local validation of the project itself:

```bash
make test
make dry-run
```

## Decision rule after the pilot

Proceed to the 300M-token run when training is numerically stable, held-out recipe perplexity improves, and at least one held-out food metric improves without a large replay-perplexity or MMLU Nutrition regression. If recipe metrics improve but replay perplexity degrades materially, compare the current 5% replay setting with a 10% replay ablation before the corpus pass. Do not choose the final checkpoint from training loss alone.
