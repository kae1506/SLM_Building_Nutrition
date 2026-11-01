import ast
import inspect
from pathlib import Path

from transformers import TrainingArguments


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_training_arguments_match_installed_transformers() -> None:
    tree = ast.parse((PROJECT_ROOT / "src" / "recipe_dapt" / "train.py").read_text())
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and getattr(node.func, "id", None) == "TrainingArguments"
    ]
    assert len(calls) == 1
    supplied = {keyword.arg for keyword in calls[0].keywords}
    supported = set(inspect.signature(TrainingArguments).parameters)
    assert supplied <= supported
