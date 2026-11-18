"""Compare Bluetooth attributions on benign records using the published checkpoints."""
import json
from pathlib import Path

import numpy as np
import shap
import torch

from src.config import SEEDS, SHAP_ATTACK_SAMPLE_SIZE, SHAP_BACKGROUND_SIZE
from src.explain import _spearman_cosine
from src.model import IoMTMLP
from src.utils import load_and_preprocess


def compare(seed):
    train, test, train_y, test_y = load_and_preprocess("bluetooth", random_state=seed)
    rng = torch.Generator().manual_seed(seed)
    train_rows = torch.where(train_y.ravel() == 0)[0]
    test_rows = torch.where(test_y.ravel() == 0)[0]
    bg = train_rows[torch.randperm(len(train_rows), generator=rng)[:SHAP_BACKGROUND_SIZE]]
    sample = test_rows[torch.randperm(len(test_rows), generator=rng)[:SHAP_ATTACK_SAMPLE_SIZE]]
    vectors = []
    for checkpoint in (f"study_fedavg_seed{seed}", f"study_fedper_seed{seed}_bluetooth"):
        model = IoMTMLP(train.shape[1])
        model.load_state_dict(torch.load(Path("models") / (checkpoint + ".pth"), weights_only=True))
        explainer = shap.DeepExplainer(model.eval(), train[bg])
        phi = np.asarray(explainer.shap_values(test[sample])).reshape(len(sample), train.shape[1])
        vectors.append(abs(phi).mean(axis=0))
    rho, cosine = _spearman_cosine(*vectors)
    a, b = (set(np.argsort(-v, kind="stable")[:5]) for v in vectors)
    return {"seed": seed, "test_positions": sample.tolist(), "background_train_positions": bg.tolist(),
            "spearman": rho, "cosine": cosine, "jaccard": len(a & b) / len(a | b)}


if __name__ == "__main__":
    result = [compare(seed) for seed in SEEDS]
    Path("results/benign_shap_probe.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
