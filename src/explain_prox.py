"""SHAP explanation generation and cross-client divergence analysis for FedProx model."""

import os
import sys

# Ensure numba JIT does not trigger Windows Application Control DLL block
os.environ["NUMBA_DISABLE_JIT"] = "1"

# Allow running as `python src/explain_prox.py` from repo root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import DATA_DIR, MODELS_DIR
from src.explain import compute_shap_divergence


def main() -> None:
    model_path = os.path.join(MODELS_DIR, "fedprox_global.pth")
    compute_shap_divergence(
        model_or_path=model_path,
        data_dir=DATA_DIR,
        plot_prefix="fedprox",
        title_suffix="FedProx Global Model",
        save_plots=True,
    )


if __name__ == "__main__":
    main()
