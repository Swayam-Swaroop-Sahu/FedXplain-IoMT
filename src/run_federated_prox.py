"""Federated FedProx orchestration with per-protocol IoMT clients."""

import os
import sys

import numpy as np
import torch

# Allow running as `python src/run_federated_prox.py` from repo root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.client import IoMTFlowerClient
from src.config import (
    DATA_DIR,
    MODELS_DIR,
    MU_VALUES,
    N_LOCAL_EPOCHS,
    N_ROUNDS,
    PROTOCOLS,
)
from src.model import IoMTMLP
from src.run_federated import fedavg_weighted
from src.utils import get_input_dim, set_seed


def train_fedprox(
    data_dir: str = DATA_DIR,
    seed: int = 42,
    mu: float = 0.01,
    n_rounds: int = N_ROUNDS,
    n_epochs: int = N_LOCAL_EPOCHS,
    save_path: str | None = None,
) -> dict:
    """Run FedProx training loop across all clients with proximal regularization."""
    set_seed(seed)

    input_dim = get_input_dim("wifi", data_dir=data_dir)

    # Guardrail: verify all protocols produce the same feature count
    for proto in PROTOCOLS:
        proto_dim = get_input_dim(proto, data_dir=data_dir)
        assert proto_dim == input_dim, (
            f"input_dim mismatch: '{proto}' has {proto_dim} features but "
            f"expected {input_dim}. Preprocessing may have changed."
        )

    # Fresh model for initial global weights
    global_model = IoMTMLP(input_dim)
    global_weights = global_model.get_weights()

    history: list[dict[str, dict]] = []

    print(f"Starting FedProx (mu={mu}): {n_rounds} rounds, {n_epochs} local epochs, "
          f"{len(PROTOCOLS)} clients, seed={seed}, data_dir={data_dir}\n")

    for rnd in range(1, n_rounds + 1):
        print(f"--- Round {rnd}/{n_rounds} (mu={mu}) ---")

        # ----- FIT phase with proximal regularization -----
        fit_results: list[tuple[list[np.ndarray], int]] = []
        for cid, proto in enumerate(PROTOCOLS):
            client = IoMTFlowerClient(cid=cid, protocol=proto, data_dir=data_dir, seed=seed)
            updated_weights, n_train, metrics = client.fit(
                global_weights, {"mu": mu, "epochs": n_epochs}
            )
            fit_results.append((updated_weights, n_train))

        # ----- AGGREGATE (weighted FedAvg / FedProx aggregation) -----
        global_weights = fedavg_weighted(fit_results)

        # ----- EVALUATE phase -----
        round_metrics: dict[str, dict] = {}
        for cid, proto in enumerate(PROTOCOLS):
            client = IoMTFlowerClient(cid=cid, protocol=proto, data_dir=data_dir, seed=seed)
            loss, n_test, metrics = client.evaluate(global_weights, {})
            round_metrics[proto] = metrics
            print(
                f"  Eval {proto:<10s} - Loss: {loss:.4f} | "
                f"Binary F1: {metrics['f1_binary']:.4f} | Macro F1: {metrics['f1_macro']:.4f} | "
                f"Benign Prec: {metrics['benign_precision']:.4f} | Attack Rec: {metrics['attack_recall']:.4f}"
            )

        avg_macro_f1 = np.mean([m["f1_macro"] for m in round_metrics.values()])
        avg_bin_f1 = np.mean([m["f1_binary"] for m in round_metrics.values()])
        print(f"  Round {rnd} Summary -> Avg Macro F1: {avg_macro_f1:.4f} | Avg Binary F1: {avg_bin_f1:.4f}\n")
        history.append(round_metrics)

    # ------------------------------------------------------------------
    # Save global model
    # ------------------------------------------------------------------
    if save_path is None:
        os.makedirs(MODELS_DIR, exist_ok=True)
        suffix = f"_mu{mu}_seed{seed}.pth" if (seed != 42 or mu != 0.01) else ".pth"
        save_path = os.path.join(MODELS_DIR, f"fedprox_global{suffix}")

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    final_model = IoMTMLP(input_dim)
    final_model.set_weights(global_weights)
    torch.save(final_model.state_dict(), save_path)

    # ------------------------------------------------------------------
    # Summary table
    # ------------------------------------------------------------------
    print("=" * 82)
    print(f"{'Round':>5} | {'Wi-Fi (Macro/Bin)':>18} | {'MQTT (Macro/Bin)':>18} | {'BT (Macro/Bin)':>18} | {'Avg Macro':>10}")
    print("-" * 82)
    for rnd, r_metrics in enumerate(history, 1):
        w = r_metrics["wifi"]
        m = r_metrics["mqtt"]
        b = r_metrics["bluetooth"]
        avg_mac = np.mean([w["f1_macro"], m["f1_macro"], b["f1_macro"]])
        print(
            f"{rnd:>5} | {w['f1_macro']:>7.4f}/{w['f1_binary']:<7.4f} | "
            f"{m['f1_macro']:>7.4f}/{m['f1_binary']:<7.4f} | "
            f"{b['f1_macro']:>7.4f}/{b['f1_binary']:<7.4f} | {avg_mac:>10.4f}"
        )
    print("=" * 82)

    final_metrics = history[-1]
    return {
        "aggregator": "fedprox",
        "mu": mu,
        "seed": seed,
        "data_dir": data_dir,
        "n_rounds": n_rounds,
        "n_local_epochs": n_epochs,
        "model_path": save_path,
        "client_metrics": final_metrics,
        "history": history,
    }


def main() -> None:
    train_fedprox()


if __name__ == "__main__":
    main()
