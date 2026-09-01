"""Centralized configuration for FedXplain-IoMT experiments."""

import os

# ----------------------------------------------------------------------
# Experiment scale & hyperparameters
# ----------------------------------------------------------------------
N_ROUNDS: int = 10
N_LOCAL_EPOCHS: int = 3
SEEDS: list[int] = [42, 7]
MU_VALUES: list[float] = [0.01, 0.1]

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
