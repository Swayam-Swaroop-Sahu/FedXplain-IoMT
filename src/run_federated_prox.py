"""Federated FedProx demo: manual round orchestration with 3 protocol clients."""

import os
import sys

import numpy as np
import torch

# Allow running as `python src/run_federated_prox.py` from repo root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import set_seed, get_input_dim
from src.model import IoMTMLP
from src.client import IoMTFlowerClient
from src.run_federated import fedavg_weighted

PROTOCOLS = ["wifi", "mqtt", "bluetooth"]
NUM_ROUNDS = 5
MU = 0.01


def main() -> None:
    set_seed(42)

    # Determine input dimension (same across protocols after preprocessing)
    input_dim = get_input_dim("wifi")

    # Guardrail: verify all protocols produce the same feature count
    for proto in PROTOCOLS:
        proto_dim = get_input_dim(proto)
        assert proto_dim == input_dim, (
            f"input_dim mismatch: '{proto}' has {proto_dim} features but "
            f"expected {input_dim}. Preprocessing may have changed."
        )

    # Fresh model for initial global weights
    global_model = IoMTMLP(input_dim)
    global_weights = global_model.get_weights()

    # Track per-round evaluation F1s for the summary table
    history: list[dict[str, float]] = []

    print(f"Starting FedProx (mu={MU}): {NUM_ROUNDS} rounds, 2 local epochs, 3 clients\n")

    for rnd in range(1, NUM_ROUNDS + 1):
        print(f"--- Round {rnd}/{NUM_ROUNDS} ---")

        # ----- FIT phase with proximal regularization -----
        fit_results: list[tuple[list[np.ndarray], int]] = []
        for cid, proto in enumerate(PROTOCOLS):
            client = IoMTFlowerClient(cid=cid, protocol=proto)
            updated_weights, n_train, metrics = client.fit(
                global_weights, {"mu": MU}
            )
            fit_results.append((updated_weights, n_train))

        # ----- AGGREGATE (weighted FedAvg / FedProx aggregation) -----
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
    save_path = "models/fedprox_global.pth"
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
    print("Federated FedProx demo complete.")


if __name__ == "__main__":
    main()
