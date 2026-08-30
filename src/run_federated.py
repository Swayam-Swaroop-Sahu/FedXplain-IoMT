"""Federated FedAvg demo: manual round orchestration with 3 protocol clients."""

import os
import sys

import numpy as np
import torch

# Allow running as `python src/run_federated.py` from repo root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import set_seed, get_input_dim
from src.model import IoMTMLP
from src.client import IoMTFlowerClient

PROTOCOLS = ["wifi", "mqtt", "bluetooth"]
NUM_ROUNDS = 5


def fedavg_weighted(results: list[tuple[list[np.ndarray], int]]) -> list[np.ndarray]:
    """Weighted average of model weights (FedAvg)."""
    total_samples = sum(n for _, n in results)
    averaged = []
    num_layers = len(results[0][0])
    for layer_idx in range(num_layers):
        weighted_sum = np.zeros_like(results[0][0][layer_idx])
        for weights, n_samples in results:
            weighted_sum += weights[layer_idx] * (n_samples / total_samples)
        averaged.append(weighted_sum)
    return averaged


def main() -> None:
    set_seed(42)

    # Determine input dimension (same across protocols after preprocessing)
    input_dim = get_input_dim("wifi")

    # Fresh model for initial global weights
    global_model = IoMTMLP(input_dim)
    global_weights = global_model.get_weights()

    # Track per-round evaluation F1s for the summary table
    history: list[dict[str, float]] = []

    print(f"Starting FedAvg: {NUM_ROUNDS} rounds, 2 local epochs, 3 clients\n")

    for rnd in range(1, NUM_ROUNDS + 1):
        print(f"--- Round {rnd}/{NUM_ROUNDS} ---")

        # ----- FIT phase -----
        fit_results: list[tuple[list[np.ndarray], int]] = []
        for cid, proto in enumerate(PROTOCOLS):
            client = IoMTFlowerClient(cid=cid, protocol=proto)
            updated_weights, n_train, metrics = client.fit(global_weights, {})
            fit_results.append((updated_weights, n_train))

        # ----- AGGREGATE (weighted FedAvg) -----
        global_weights = fedavg_weighted(fit_results)

        # ----- EVALUATE phase -----
        round_f1s: dict[str, float] = {}
        for cid, proto in enumerate(PROTOCOLS):
            client = IoMTFlowerClient(cid=cid, protocol=proto)
            loss, n_test, metrics = client.evaluate(global_weights, {})
            round_f1s[proto] = metrics["f1"]
            print(f"  Eval {proto:<10s} - Loss: {loss:.4f}  F1: {metrics['f1']:.4f}")

        avg_f1 = np.mean(list(round_f1s.values()))
        print(f"  Average F1: {avg_f1:.4f}\n")
        history.append(round_f1s)

    # ------------------------------------------------------------------
    # Save global model
    # ------------------------------------------------------------------
    os.makedirs("models", exist_ok=True)
    final_model = IoMTMLP(input_dim)
    final_model.set_weights(global_weights)
    save_path = "models/fedavg_global.pth"
    torch.save(final_model.state_dict(), save_path)

    # ------------------------------------------------------------------
    # Summary table
    # ------------------------------------------------------------------
    print("=" * 62)
    print(f"{'Round':>5} | {'Wi-Fi F1':>9} | {'MQTT F1':>8} | {'BT F1':>8} | {'Avg F1':>8}")
    print("-" * 62)
    for rnd, f1s in enumerate(history, 1):
        avg = np.mean(list(f1s.values()))
        print(
            f"{rnd:>5} | {f1s['wifi']:>9.4f} | {f1s['mqtt']:>8.4f} | "
            f"{f1s['bluetooth']:>8.4f} | {avg:>8.4f}"
        )
    print("=" * 62)

    # Model size check
    size_kb = os.path.getsize(save_path) / 1024
    print(f"\nModel saved to {save_path}  ({size_kb:.1f} KB)")
    print("Federated FedAvg demo complete.")


if __name__ == "__main__":
    main()
