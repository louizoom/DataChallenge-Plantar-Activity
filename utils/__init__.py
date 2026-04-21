"""
Shared utilities for the Plantar Activity Classification project.

Exports
-------
Data loading & windowing functions:
    load_and_merge_data, create_windows, create_windows_with_ids,
    load_all_subjects, clean_dataframe

Path constants:
    PROJECT_ROOT, BASE_DIR, PLANTAR_DIR, EVENTS_DIR,
    OUTPUTS_DIR, MODELS_DIR, RESULTS_DIR
"""

from .data_utils import (
    load_and_merge_data,
    create_windows,
    create_windows_with_ids,
    load_all_subjects,
    clean_dataframe,
)
from .paths import (
    PROJECT_ROOT,
    BASE_DIR,
    PLANTAR_DIR,
    EVENTS_DIR,
    OUTPUTS_DIR,
    MODELS_DIR,
    RESULTS_DIR,
    RESULTS_CHARTS_DIR,
    RESULTS_LOGS_DIR,
    RESULTS_METRICS_DIR,
)

__all__ = [
    # Data utilities
    "load_and_merge_data",
    "create_windows",
    "create_windows_with_ids",
    "load_all_subjects",
    "clean_dataframe",
    # Path constants
    "PROJECT_ROOT",
    "BASE_DIR",
    "PLANTAR_DIR",
    "EVENTS_DIR",
    "OUTPUTS_DIR",
    "MODELS_DIR",
    "RESULTS_DIR",
    "RESULTS_CHARTS_DIR",
    "RESULTS_LOGS_DIR",
    "RESULTS_METRICS_DIR",
]
