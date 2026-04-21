"""
Shared data loading and preprocessing utilities for the Plantar Activity Classification project.

All training scripts import from this module to avoid code duplication.

Functions
---------
load_and_merge_data     : Load and temporally align a single subject/sequence pair.
create_windows          : Sliding-window segmentation (returns X, y).
create_windows_with_ids : Sliding-window segmentation with subject-group labels (for GroupKFold).
load_all_subjects       : Load and concatenate data for all available subjects.
clean_dataframe         : Drop un-labelled rows and fill missing values.
"""

import os
from collections import Counter
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from .paths import PLANTAR_DIR, EVENTS_DIR


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------

def load_and_merge_data(subject_id, sequence, sep: str = ";") -> Optional[pd.DataFrame]:
    """
    Load and temporally align sensor data with class annotations for one sequence.

    The insoles CSV (100 FPS sensor readings) is merged with the classification
    CSV (action labels with start/end timestamps) by assigning each sensor frame
    the label of the action that was active at that timestamp.

    Parameters
    ----------
    subject_id : str | int
        Subject identifier, e.g. ``'01'`` or ``1``.
    sequence : str
        Sequence folder name, e.g. ``'Sequence_01'``.
    sep : str, optional
        CSV delimiter used in both files. Defaults to ``';'``.

    Returns
    -------
    Optional[pd.DataFrame]
        Merged DataFrame with a ``'Class'`` column, or ``None`` if files are
        missing.
    """
    if isinstance(subject_id, int):
        subject_id = f"{subject_id:02d}"

    insoles_path = os.path.join(PLANTAR_DIR, f"S{subject_id}", sequence, "insoles.csv")
    classif_path = os.path.join(EVENTS_DIR, f"S{subject_id}", sequence, "classif.csv")

    if not os.path.exists(insoles_path) or not os.path.exists(classif_path):
        return None

    df_insoles = pd.read_csv(insoles_path, sep=sep)
    df_classif = pd.read_csv(classif_path, sep=sep)

    df_insoles["Class"] = np.nan
    for _, row in df_classif.iterrows():
        mask = (df_insoles["Time"] >= row["Timestamp Start"]) & (
            df_insoles["Time"] <= row["Timestamp End"]
        )
        df_insoles.loc[mask, "Class"] = row["Class"]

    return df_insoles


# ---------------------------------------------------------------------------
# Windowing
# ---------------------------------------------------------------------------

def create_windows(
    X: np.ndarray,
    y: np.ndarray,
    window_size: int = 50,
    step_size: int = 25,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Segment time-series data into overlapping windows.

    The class label for each window is determined by majority vote over the
    frames within that window.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix of shape ``(N, num_features)``.
    y : np.ndarray
        Label array of shape ``(N,)``.
    window_size : int, optional
        Number of frames per window. Defaults to ``50`` (~0.5 s at 100 FPS).
    step_size : int, optional
        Stride between consecutive windows. Defaults to ``25`` (50 % overlap).

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        ``(X_windows, y_windows)`` with shapes
        ``(num_windows, window_size, num_features)`` and ``(num_windows,)``.
    """
    windows_X, windows_y = [], []
    for i in range(0, len(X) - window_size, step_size):
        win_X = X[i : i + window_size]
        win_y = y[i : i + window_size]
        maj_y = Counter(win_y).most_common(1)[0][0]
        windows_X.append(win_X)
        windows_y.append(maj_y)
    return np.array(windows_X), np.array(windows_y)


def create_windows_with_ids(
    X: np.ndarray,
    y: np.ndarray,
    subject_id: int,
    window_size: int = 50,
    step_size: int = 25,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Like :func:`create_windows`, but also returns per-window subject identifiers.

    The subject IDs are used as group labels for ``GroupKFold`` cross-validation,
    ensuring that windows from the same patient never appear in both the training
    and validation folds.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix of shape ``(N, num_features)``.
    y : np.ndarray
        Label array of shape ``(N,)``.
    subject_id : int
        Numeric subject identifier (e.g. ``1`` for S01).
    window_size : int, optional
        Number of frames per window. Defaults to ``50``.
    step_size : int, optional
        Stride between consecutive windows. Defaults to ``25``.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        ``(X_windows, y_windows, subject_ids)`` — all of length ``num_windows``.
    """
    windows_X, windows_y, windows_ids = [], [], []
    for i in range(0, len(X) - window_size, step_size):
        win_X = X[i : i + window_size]
        win_y = y[i : i + window_size]
        maj_y = Counter(win_y).most_common(1)[0][0]
        windows_X.append(win_X)
        windows_y.append(maj_y)
        windows_ids.append(subject_id)
    return np.array(windows_X), np.array(windows_y), np.array(windows_ids)


# ---------------------------------------------------------------------------
# Bulk Loading
# ---------------------------------------------------------------------------

def load_all_subjects(n_subjects: int = 32, verbose: bool = True) -> Optional[pd.DataFrame]:
    """
    Load and concatenate data for all available subjects.

    Subjects are iterated from S01 up to ``S{n_subjects}``. Missing subject
    directories are silently skipped so the dataset adapts to whatever subjects
    are present in the data folder.

    Parameters
    ----------
    n_subjects : int, optional
        Maximum number of subjects to load. Defaults to ``32``.
    verbose : bool, optional
        If ``True``, print a status line for each loaded sequence.

    Returns
    -------
    Optional[pd.DataFrame]
        Concatenated DataFrame, or ``None`` if no data was found.
    """
    subjects = [f"{i:02d}" for i in range(1, n_subjects + 1)]
    all_data = []
    for subj in subjects:
        subj_dir = os.path.join(PLANTAR_DIR, f"S{subj}")
        if not os.path.isdir(subj_dir):
            continue
        for seq in sorted(os.listdir(subj_dir)):
            if seq.startswith("Sequence_"):
                df = load_and_merge_data(subj, seq)
                if df is not None:
                    all_data.append(df)
                    if verbose:
                        print(f"  ✅ S{subj} — {seq} loaded.")
    if not all_data:
        return None
    return pd.concat(all_data, ignore_index=True)


# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------

def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove un-labelled rows and fill any remaining NaN values.

    Frames without a class annotation (transition periods between two actions)
    are dropped. Residual NaNs in sensor columns are filled by forward- then
    backward-propagation.

    Parameters
    ----------
    df : pd.DataFrame
        Raw merged DataFrame produced by :func:`load_and_merge_data`.

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame with no NaN values in the ``'Class'`` column.
    """
    df = df.dropna(subset=["Class"])
    df = df.ffill().bfill()
    return df
