from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"

FINAL_RESULTS_DIR = RESULTS_DIR / "final_paper"
UNCERTAINTY_RESULTS_DIR = RESULTS_DIR / "uncertainty"
OOD_RESULTS_DIR = RESULTS_DIR / "ood"
METRICS_RESULTS_DIR = RESULTS_DIR / "metrics"

for directory in [
    RESULTS_DIR,
    FINAL_RESULTS_DIR,
    UNCERTAINTY_RESULTS_DIR,
    OOD_RESULTS_DIR,
    METRICS_RESULTS_DIR,
]:
    directory.mkdir(
        parents=True,
        exist_ok=True
    )
