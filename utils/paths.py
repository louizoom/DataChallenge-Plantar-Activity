"""
Centralised path configuration for the Plantar Activity Classification project.

All scripts must import paths from this module to guarantee consistency.
Paths are resolved from environment variables defined in the `.env` file at
the project root. Copy `.env.example` to `.env` and fill in your local values.

Environment variables
---------------------
DATA_ROOT       : Path to the data root directory (absolute or relative to project root).
                  Default: "DataChallenge_donneesGlobales"
PLANTAR_FOLDER  : Sub-folder name containing the insole CSV files.
                  Use "Plantar_activity_trie" for the sorted dataset,
                  or "Plantar_activity" for the unsorted variant.
                  Default: "Plantar_activity_trie"
EVENTS_FOLDER   : Sub-folder name containing the classification annotation files.
                  Default: "Events"
"""

import os
import sys

# Load .env file if python-dotenv is available (optional but recommended).
try:
    from dotenv import load_dotenv, find_dotenv
    _dotenv_path = find_dotenv(usecwd=True)
    if _dotenv_path:
        load_dotenv(_dotenv_path)
    else:
        # Fallback: look for .env next to this file's project root
        _project_root_env = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
        if os.path.exists(_project_root_env):
            load_dotenv(_project_root_env)
except ImportError:
    pass  # python-dotenv not installed; rely on environment variables already set

# ---------------------------------------------------------------------------
# Project root (the directory that contains `utils/`, `src/`, `results/`, …)
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ---------------------------------------------------------------------------
# Data directories — configurable via .env
# ---------------------------------------------------------------------------
_data_root_raw = os.environ.get("DATA_ROOT", "DataChallenge_donneesGlobales")

# Support both absolute and relative paths
if os.path.isabs(_data_root_raw):
    BASE_DIR = _data_root_raw
else:
    BASE_DIR = os.path.join(PROJECT_ROOT, _data_root_raw)

PLANTAR_FOLDER = os.environ.get("PLANTAR_FOLDER", "Plantar_activity_trie")
EVENTS_FOLDER = os.environ.get("EVENTS_FOLDER", "Events")

PLANTAR_DIR = os.path.join(BASE_DIR, PLANTAR_FOLDER)
EVENTS_DIR = os.path.join(BASE_DIR, EVENTS_FOLDER)

# ---------------------------------------------------------------------------
# Output directories (created automatically on import)
# ---------------------------------------------------------------------------
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs")
MODELS_DIR  = os.path.join(PROJECT_ROOT, "models")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

# results/ sub-directories — all scripts must use these instead of raw RESULTS_DIR
RESULTS_CHARTS_DIR  = os.path.join(RESULTS_DIR, "charts")   # PNG charts / figures
RESULTS_LOGS_DIR    = os.path.join(RESULTS_DIR, "logs")     # Training log files (.log)
RESULTS_METRICS_DIR = os.path.join(RESULTS_DIR, "metrics")  # JSON results & matrices

for _d in (OUTPUTS_DIR, MODELS_DIR,
           RESULTS_DIR, RESULTS_CHARTS_DIR, RESULTS_LOGS_DIR, RESULTS_METRICS_DIR):
    os.makedirs(_d, exist_ok=True)

# ---------------------------------------------------------------------------
# Startup validation — warn early if data folders are missing
# ---------------------------------------------------------------------------
def _warn_missing(path: str, name: str) -> None:
    if not os.path.isdir(path):
        print(
            f"[paths.py] WARNING: {name} directory not found: {path}\n"
            f"  → Check your .env file (copy .env.example → .env and fill in DATA_ROOT, "
            f"PLANTAR_FOLDER, EVENTS_FOLDER).",
            file=sys.stderr,
        )

_warn_missing(BASE_DIR, "DATA_ROOT")
_warn_missing(PLANTAR_DIR, "PLANTAR_DIR")
_warn_missing(EVENTS_DIR, "EVENTS_DIR")
