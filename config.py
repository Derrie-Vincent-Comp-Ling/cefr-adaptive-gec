"""Project-wide configuration for dissertation_gec.

Error-Type-Aware, CEFR-Adaptive Feedback for L2 Writing.
"""
from __future__ import annotations

from pathlib import Path

# Reproducibility
SEED: int = 42

# Root paths
PROJECT_ROOT: Path = Path(__file__).resolve().parent

# Data
DATA_DIR: Path = PROJECT_ROOT / "data"
DATA_RAW_DIR: Path = DATA_DIR / "raw"
DATA_PROCESSED_DIR: Path = DATA_DIR / "processed"

# Code modules (one concern per folder)
MODELS_DIR: Path = PROJECT_ROOT / "models"
SELECTOR_DIR: Path = PROJECT_ROOT / "selector"
ANNOTATE_DIR: Path = PROJECT_ROOT / "annotate"
FEEDBACK_DIR: Path = PROJECT_ROOT / "feedback"
FEEDBACK_TEMPLATES_DIR: Path = FEEDBACK_DIR / "templates"
PRACTICE_DIR: Path = PROJECT_ROOT / "practice"
EVAL_DIR: Path = PROJECT_ROOT / "eval"
APP_DIR: Path = PROJECT_ROOT / "app"

# Experiment + output paths
EXPERIMENTS_DIR: Path = PROJECT_ROOT / "experiments"
RESULTS_DIR: Path = PROJECT_ROOT / "results"
RESULTS_PLOTS_DIR: Path = RESULTS_DIR / "plots"
LOGS_DIR: Path = PROJECT_ROOT / "logs"
HUMAN_EVAL_DIR: Path = PROJECT_ROOT / "human_eval"

ALL_DIRS = [
    DATA_RAW_DIR,
    DATA_PROCESSED_DIR,
    MODELS_DIR,
    SELECTOR_DIR,
    ANNOTATE_DIR,
    FEEDBACK_DIR,
    FEEDBACK_TEMPLATES_DIR,
    PRACTICE_DIR,
    EVAL_DIR,
    APP_DIR,
    EXPERIMENTS_DIR,
    RESULTS_DIR,
    RESULTS_PLOTS_DIR,
    LOGS_DIR,
    HUMAN_EVAL_DIR,
]


def ensure_dirs() -> None:
    """Create all project directories if they don't exist."""
    for d in ALL_DIRS:
        d.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    ensure_dirs()
    print(f"PROJECT_ROOT = {PROJECT_ROOT}")
    print(f"SEED = {SEED}")
