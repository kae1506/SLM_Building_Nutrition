"""
config.py

Loads all configuration files and stores scoring parameters.
"""

from pathlib import Path

CONFIG_DIR = Path("config")

def load_txt(filename):
    """
    Load a text file into a Python set.

    Empty lines are ignored.
    """

    path = CONFIG_DIR / filename

    with open(path, "r", encoding="utf-8") as f:
        return {
            line.strip()
            for line in f
            if line.strip()
        }

STRONG_MESH = load_txt("strong_nutrition_mesh.txt")

CONTEXT_MESH = load_txt("context_mesh.txt")

NUTRITION_KEYWORDS = load_txt("nutrition_keywords.txt")

NUTRITION_JOURNALS = load_txt("nutrition_journals.txt")

REMOVE_PUBLICATION_TYPES = load_txt(
    "remove_publication_types.txt"
)


STRONG_MESH_SCORE = 3

CONTEXT_MESH_SCORE = 1

TITLE_KEYWORD_SCORE = 2

ABSTRACT_KEYWORD_SCORE = 1

JOURNAL_SCORE = 2

KEEP_THRESHOLD = 5

BORDERLINE_THRESHOLD = 3