import os
from pathlib import Path

from recipe_dapt.config import load_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_short_pilot_budget_and_steps() -> None:
    config = load_config(PROJECT_ROOT / "configs" / "short_pilot.yaml")
    assert config.resolved_token_budget() == 50_000_000
    assert config.tokens_per_step() == 262_144
    assert config.max_steps() == 191


def test_gradient_accumulation_scales_with_world_size() -> None:
    config = load_config(PROJECT_ROOT / "configs" / "short_pilot.yaml")
    assert config.gradient_accumulation_steps(world_size=1) == 8
    assert config.gradient_accumulation_steps(world_size=8) == 1


def test_remote_storage_overrides() -> None:
    previous_data = os.environ.get("DAPT_DATA_ROOT")
    previous_output = os.environ.get("DAPT_OUTPUT_ROOT")
    try:
        os.environ["DAPT_DATA_ROOT"] = "/tmp/recipe-dapt-test-data"
        os.environ["DAPT_OUTPUT_ROOT"] = "/tmp/recipe-dapt-test-runs"
        config = load_config(PROJECT_ROOT / "configs" / "short_pilot.yaml")
        assert config.paths["recipe_source"] == Path(
            "/tmp/recipe-dapt-test-data/processed/dapt-parquet"
        ).resolve()
        assert config.output_dir == Path("/tmp/recipe-dapt-test-runs/short_pilot").resolve()
    finally:
        if previous_data is None:
            os.environ.pop("DAPT_DATA_ROOT", None)
        else:
            os.environ["DAPT_DATA_ROOT"] = previous_data
        if previous_output is None:
            os.environ.pop("DAPT_OUTPUT_ROOT", None)
        else:
            os.environ["DAPT_OUTPUT_ROOT"] = previous_output
