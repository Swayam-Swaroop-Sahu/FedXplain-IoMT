"""Shared utilities: seeding, data loading, preprocessing, and feature inspection."""

import os
import random
import re

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.config import (
    DATA_DIR,
    EXPECTED_FEATURE_COUNT,
    TEST_SIZE,
)


def set_seed(seed: int = 42) -> None:
    """Fix random seeds for Python, NumPy and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _is_drop_column(col_name: str) -> bool:
    """Return True if the column should be dropped (ID / IP / timestamp / label)."""
    lower = col_name.strip().lower()
    # Exact matches first
    if lower in {"label", "labels"}:
        return True
    # Substring patterns for ID, IP, timestamp columns
    patterns = [
        "flow id", "flow_id", "src ip", "src_ip", "dst ip", "dst_ip",
        "timestamp", "time_stamp"
    ]
    for pat in patterns:
        if pat in lower:
            return True
    return False


def _find_csv_path(protocol: str, data_dir: str = DATA_DIR) -> str:
    """Resolve the dataset CSV path across study, poc, or raw conventions."""
    protocol = protocol.lower().strip()
    candidates = [
        os.path.join(data_dir, f"{protocol}_study.csv"),
        os.path.join(data_dir, f"{protocol}_poc.csv"),
        os.path.join(data_dir, f"{protocol}.csv"),
    ]
    for cand in candidates:
        if os.path.exists(cand):
            return cand
    raise FileNotFoundError(
        f"No CSV dataset found for protocol '{protocol}' in '{data_dir}'. "
        f"Checked candidates: {candidates}"
    )


_DATA_CACHE: dict[tuple, tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = {}


def load_and_preprocess(
    protocol: str,
    data_dir: str = DATA_DIR,
    test_size: float = TEST_SIZE,
    random_state: int = 42,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Load a protocol CSV and return preprocessed PyTorch tensors.

    Parameters
    ----------
    protocol : str
        One of 'wifi', 'mqtt', 'bluetooth' (case-insensitive).
    data_dir : str
        Directory containing dataset files. Defaults to config.DATA_DIR.
    test_size : float
        Fraction of data reserved for testing. Defaults to config.TEST_SIZE.
    random_state : int
        Seed for the stratified train/test split. Defaults to 42.

    Returns
    -------
    (X_train, X_test, y_train, y_test) as PyTorch tensors.
        X tensors are FloatTensor; y tensors are FloatTensor with shape (N, 1).
    """
    cache_key = (protocol.lower().strip(), os.path.abspath(data_dir), test_size, random_state)
    if cache_key in _DATA_CACHE:
        X_tr, X_te, y_tr, y_te = _DATA_CACHE[cache_key]
        return X_tr.clone(), X_te.clone(), y_tr.clone(), y_te.clone()

    path = _find_csv_path(protocol, data_dir)
    df = pd.read_csv(path)

    # Identify columns to drop (IDs, IPs, timestamps, label)
    drop_cols = [c for c in df.columns if _is_drop_column(c)]

    # Separate target before dropping
    label_col = [c for c in df.columns if c.strip().lower() in {"label", "labels"}]
    if not label_col:
        raise ValueError(f"No label column found in {path}")
    label_col = label_col[0]

    # Build binary target: Benign -> 0, everything else -> 1
    y = df[label_col].apply(lambda v: 0 if str(v).strip().lower() == "benign" else 1).values

    # Drop non-feature columns
    X = df.drop(columns=drop_cols, errors="ignore")

    # Safety: drop any remaining non-numeric columns
    non_numeric = X.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric:
        X = X.drop(columns=non_numeric)

    X = X.values.astype(np.float32)

    # Guardrail: assert feature count matches expected schema
    actual_features = X.shape[1]
    assert actual_features == EXPECTED_FEATURE_COUNT, (
        f"Feature count mismatch for protocol '{protocol}': expected "
        f"{EXPECTED_FEATURE_COUNT}, got {actual_features}. This indicates a "
        f"preprocessing or CSV schema change that will produce incomparable "
        f"results across runs. Audit _is_drop_column() and the input CSV."
    )

    # Replace NaN / Inf with 0
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    # Train / test split (stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    # Standard-scale (fit on train only)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_test = scaler.transform(X_test).astype(np.float32)

    # Replace any post-scaling NaN/Inf
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
    X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)

    # Convert to PyTorch tensors
    X_train_t = torch.FloatTensor(X_train)
    X_test_t = torch.FloatTensor(X_test)
    y_train_t = torch.FloatTensor(y_train.astype(np.float32)).unsqueeze(1)
    y_test_t = torch.FloatTensor(y_test.astype(np.float32)).unsqueeze(1)

    _DATA_CACHE[cache_key] = (X_train_t, X_test_t, y_train_t, y_test_t)
    return X_train_t.clone(), X_test_t.clone(), y_train_t.clone(), y_test_t.clone()


def get_input_dim(protocol: str = "wifi", data_dir: str = DATA_DIR) -> int:
    """Return the number of feature columns after dropping IDs/labels."""
    path = _find_csv_path(protocol, data_dir)
    df = pd.read_csv(path, nrows=5)

    drop_cols = [c for c in df.columns if _is_drop_column(c)]
    X = df.drop(columns=drop_cols, errors="ignore")
    non_numeric = X.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric:
        X = X.drop(columns=non_numeric)
    return X.shape[1]


def get_feature_names(protocol: str = "wifi", data_dir: str = DATA_DIR) -> list[str]:
    """Return list of feature column names after dropping non-features."""
    path = _find_csv_path(protocol, data_dir)
    df = pd.read_csv(path, nrows=5)

    drop_cols = [c for c in df.columns if _is_drop_column(c)]
    X = df.drop(columns=drop_cols, errors="ignore")
    non_numeric = X.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric:
        X = X.drop(columns=non_numeric)
    return list(X.columns)
