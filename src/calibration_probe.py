"""Retrospective Bluetooth Platt probe. Run python -m src.calibration_probe."""
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.calibration import calibration_curve
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, confusion_matrix, log_loss, roc_auc_score
from sklearn.model_selection import train_test_split

from src.config import SEEDS
from src.model import IoMTMLP
from src.utils import load_and_preprocess


def scores(labels, probabilities):
    observed, predicted = calibration_curve(labels, probabilities, n_bins=10, strategy="quantile")
    tn, fp, fn, tp = confusion_matrix(labels, probabilities >= 0.5, labels=[0, 1]).ravel()
    return {
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "benign_recall": tn / (tn + fp), "attack_recall": tp / (tp + fn),
        "brier": brier_score_loss(labels, probabilities), "log_loss": log_loss(labels, probabilities),
        "roc_auc": roc_auc_score(labels, probabilities),
        "reliability": {"mean_score": predicted.tolist(), "attack_fraction": observed.tolist()},
    }


def probe(seed, reference):
    _, features, _, labels = load_and_preprocess("bluetooth", random_state=seed)
    labels = labels.numpy().ravel()
    model = IoMTMLP(features.shape[1])
    path = Path("models") / f"study_fedavg_seed{seed}.pth"
    model.load_state_dict(torch.load(path, weights_only=True, map_location="cpu"))
    model.eval()
    with torch.inference_mode():
        logits = model.net[:-1](features).double().numpy().reshape(-1, 1)
    baseline = scores(labels, expit(logits.ravel()))
    if any(baseline[k] != reference[k] for k in ("tn", "fp", "fn", "tp")):
        raise ValueError(f"Checkpoint disagrees with published confusion matrix for seed {seed}")
    fit, test = train_test_split(np.arange(len(labels)), stratify=labels, random_state=seed, test_size=0.5)
    scaler = LogisticRegression(C=1.0, solver="lbfgs").fit(logits[fit], labels[fit])
    return {
        "seed": seed, "checkpoint": path.as_posix(), "fit_rows": len(fit), "evaluation_rows": len(test),
        "a": scaler.coef_.item(), "b": scaler.intercept_.item(),
        "baseline": scores(labels[test], expit(logits[test].ravel())),
        "platt": scores(labels[test], scaler.predict_proba(logits[test])[:, 1]),
    }


if __name__ == "__main__":
    study = json.loads(Path("results/study_results.json").read_text())
    refs = {r["seed"]: r["client_metrics"]["bluetooth"] for r in study["runs"] if r["aggregator"] == "fedavg"}
    result = {
        "protocol": "Original Bluetooth test partition retrospectively halved by class for fitting and evaluation. Both halves were held out of base training. Threshold 0.5.",
        "runs": [probe(seed, refs[seed]) for seed in SEEDS],
    }
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(SEEDS), figsize=(7.5, 3.6), sharey=True)
    for ax, run in zip(axes, result["runs"]):
        ax.plot([0, 1], [0, 1], "--", color="0.6", label="Ideal")
        for key, label in [("baseline", "FedAvg"), ("platt", "Platt")]:
            curve = run[key]["reliability"]
            ax.plot(curve["mean_score"], curve["attack_fraction"], "o-", markersize=4, label=label)
        ax.set(title=f"Seed {run['seed']}", xlabel="Mean predicted attack probability", xlim=(-0.03, 1.03))
        ax.legend()
    axes[0].set_ylabel("Observed attack fraction")
    fig.tight_layout()
    fig.savefig("figures/calibration_reliability.png", dpi=300)
    plt.close(fig)
    Path("results/calibration_probe.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")

