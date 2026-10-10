"""Centralized configuration for FedXplain-IoMT experiments."""

import os

# ----------------------------------------------------------------------
# Experiment scale & hyperparameters
# ----------------------------------------------------------------------
N_ROUNDS: int = 10
N_LOCAL_EPOCHS: int = 3
SEEDS: list[int] = [42, 7]
# mu sweep extended below 0.01 to test whether a gentler proximal pull helps
# MQTT without breaking Bluetooth (0.01/0.1 already collapsed Bluetooth).
MU_VALUES: list[float] = [0.001, 0.01, 0.05, 0.1]

# FedPer: number of leading parameter arrays (as returned by
# IoMTMLP.get_weights()) that form the shared trunk and get aggregated.
# IoMTMLP exposes 6 arrays in order: net.0.weight/bias (45->64),
# net.3.weight/bias (64->32), net.5.weight/bias (32->1 output head).
# The first 4 are the trunk (shared); the last 2 are the private
# per-client output head that is never uploaded or averaged.
FEDPER_SHARED_PARAM_COUNT: int = 4

DATA_DIR: str = "data/study"
POC_DATA_DIR: str = "data/poc"

# Protocols evaluated
PROTOCOLS: list[str] = ["wifi", "mqtt", "bluetooth"]
PROTOCOL_MAP: dict[int, str] = {0: "wifi", 1: "mqtt", 2: "bluetooth"}

# Preprocessing & model architecture
EXPECTED_FEATURE_COUNT: int = 45
TEST_SIZE: float = 0.2
BATCH_SIZE: int = 64
LEARNING_RATE: float = 0.001
CENTRALIZED_EPOCHS: int = 10

# SHAP explainability parameters
SHAP_BACKGROUND_SIZE: int = 100
SHAP_ATTACK_SAMPLE_SIZE: int = 200

# Output paths
RESULTS_DIR: str = "results"
MODELS_DIR: str = "models"
STUDY_RESULTS_PATH: str = os.path.join(RESULTS_DIR, "study_results.json")
