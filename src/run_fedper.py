"""Federated FedPer orchestration: shared trunk aggregation with private per-client heads.

Shared parameters (the 45->64 and 64->32 layers) are broadcast, trained
locally, and averaged sample-count-weighted across clients each round.
The 32->1 output head stays private: never uploaded, never averaged.
Each client's final model is the global trunk plus its own head, and is
evaluated and explained per client.
"""

import json
import os
import sys
import time

import numpy as np
import torch

# Allow running as `python src/run_fedper.py` from repo root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.client import IoMTFlowerClient
from src.config import (
    DATA_DIR,
    FEDPER_SHARED_PARAM_COUNT,
    MODELS_DIR,
    N_LOCAL_EPOCHS,
    N_ROUNDS,
    PROTOCOLS,
    SEEDS,
    SHAP_ATTACK_SAMPLE_SIZE,
    SHAP_BACKGROUND_SIZE,
    STUDY_RESULTS_PATH,
)
from src.explain import _spearman_cosine, compute_pairwise_divergence, explain_client
from src.model import IoMTMLP
from src.run_federated import fedavg_weighted
from src.utils import get_feature_names, get_input_dim, set_seed


def split_weights(weights: list[np.ndarray]) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Split full model weights into (shared trunk, private head)."""
    return weights[:FEDPER_SHARED_PARAM_COUNT], weights[FEDPER_SHARED_PARAM_COUNT:]


def merge_weights(trunk: list[np.ndarray], head: list[np.ndarray]) -> list[np.ndarray]:
    """Reassemble full model weights from a shared trunk and a private head."""
    return list(trunk) + list(head)


def train_fedper(
    seed: int = 42,
    data_dir: str = DATA_DIR,
    n_rounds: int = N_ROUNDS,
    n_local_epochs: int = N_LOCAL_EPOCHS,
    save_dir: str = MODELS_DIR,
    compute_shap: bool = True,
) -> dict:
    """Run FedPer training across all clients and explain each assembled model.

    Each round: broadcast the shared trunk -> each client trains trunk + its
    private head locally with plain BCE loss (no proximal term) -> clients
    upload only the trunk -> the server averages trunks sample-count-weighted
    (fedavg_weighted from run_federated.py). After the final round, each
    client's model (global trunk + private head) is evaluated on its own
    protocol's test set and passed through the full SHAP pipeline from
    explain.py.

    Returns a dict with the same result schema as the other federated runs
    (aggregator="fedper", mu=None) plus the per-client assembled-model SHAP
    data. Cross-client divergence keys describe the *personalized* models,
    since FedPer produces one model per client.
    """
    set_seed(seed)

    input_dim = get_input_dim("wifi", data_dir=data_dir)

    # Guardrail: verify all protocols produce the same feature count
    for proto in PROTOCOLS:
        proto_dim = get_input_dim(proto, data_dir=data_dir)
        assert proto_dim == input_dim, (
            f"input_dim mismatch: '{proto}' has {proto_dim} features but "
            f"expected {input_dim}. Preprocessing may have changed."
        )

    # Fresh model for initial trunk and private heads. All clients start from
    # the same deterministic head (set_seed above makes this reproducible);
    # the heads diverge from round 1 onward through local training only.
    init_model = IoMTMLP(input_dim)
    init_weights = init_model.get_weights()
    assert 0 < FEDPER_SHARED_PARAM_COUNT < len(init_weights), (
        f"FEDPER_SHARED_PARAM_COUNT={FEDPER_SHARED_PARAM_COUNT} must select a "
        f"strict subset of the {len(init_weights)} IoMTMLP parameter arrays."
    )
    trunk = init_weights[:FEDPER_SHARED_PARAM_COUNT]
    private_heads: dict[str, list[np.ndarray]] = {
        proto: [w.copy() for w in init_weights[FEDPER_SHARED_PARAM_COUNT:]]
        for proto in PROTOCOLS
    }

    history: list[dict[str, dict]] = []

    print(f"Starting FedPer: {n_rounds} rounds, {n_local_epochs} local epochs, "
          f"{len(PROTOCOLS)} clients, seed={seed}, data_dir={data_dir}\n")

    for rnd in range(1, n_rounds + 1):
        print(f"--- Round {rnd}/{n_rounds} ---")

        # ----- FIT phase: plain BCE local training on trunk + private head -----
        fit_results: list[tuple[list[np.ndarray], int]] = []
        for cid, proto in enumerate(PROTOCOLS):
            client = IoMTFlowerClient(cid=cid, protocol=proto, data_dir=data_dir, seed=seed)
            full_weights = merge_weights(trunk, private_heads[proto])
            updated_weights, n_train, _ = client.fit(
                full_weights, {"epochs": n_local_epochs}
            )
            # Client uploads ONLY the trunk; the trained head stays private
            updated_trunk, updated_head = split_weights(updated_weights)
            private_heads[proto] = updated_head
            fit_results.append((updated_trunk, n_train))

        # ----- AGGREGATE trunks (weighted FedAvg over shared parameters) -----
        trunk = fedavg_weighted(fit_results)

        # ----- EVALUATE phase: each client's assembled trunk + private head -----
        round_metrics: dict[str, dict] = {}
        for cid, proto in enumerate(PROTOCOLS):
            client = IoMTFlowerClient(cid=cid, protocol=proto, data_dir=data_dir, seed=seed)
            loss, _, metrics = client.evaluate(
                merge_weights(trunk, private_heads[proto]), {}
            )
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
    # Assemble and save each client's final model (global trunk + private head)
    # ------------------------------------------------------------------
    os.makedirs(save_dir, exist_ok=True)
    model_paths: dict[str, str] = {}
    assembled_models: dict[str, IoMTMLP] = {}
    for proto in PROTOCOLS:
        model = IoMTMLP(input_dim)
        model.set_weights(merge_weights(trunk, private_heads[proto]))
        model.eval()
        assembled_models[proto] = model
        path = os.path.join(save_dir, f"study_fedper_seed{seed}_{proto}.pth")
        torch.save(model.state_dict(), path)
        model_paths[proto] = path

    final_metrics = history[-1]
    avg_macro_f1 = float(np.mean([m["f1_macro"] for m in final_metrics.values()]))

    # ------------------------------------------------------------------
    # Summary table (same format as the other federated runners)
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

    # ------------------------------------------------------------------
    # SHAP: full existing pipeline per assembled per-client model
    # ------------------------------------------------------------------
    result: dict = {
        "seed": seed,
        "aggregator": "fedper",
        "mu": None,
        "data_dir": data_dir,
        "n_rounds": n_rounds,
        "n_local_epochs": n_local_epochs,
        "model_paths": model_paths,
        "client_metrics": final_metrics,
        "avg_macro_f1": avg_macro_f1,
        "history": history,
    }

    if not compute_shap:
        return result

    feature_names = get_feature_names("wifi", data_dir=data_dir)
    client_top10: dict[str, list[tuple[str, float]]] = {}
    client_top5: dict[str, list[str]] = {}
    client_full_vectors: dict[str, dict[str, list]] = {}

    for proto in PROTOCOLS:
        top10, top5, full_vec = explain_client(
            assembled_models[proto],
            proto,
            feature_names,
            data_dir=data_dir,
            seed=seed,
            bg_size=SHAP_BACKGROUND_SIZE,
            attack_sample_size=SHAP_ATTACK_SAMPLE_SIZE,
        )
        client_top10[proto] = top10
        client_top5[proto] = top5
        client_full_vectors[proto] = full_vec

    jaccard_pairwise, shared_features, divergence_full = compute_pairwise_divergence(
        client_top5, client_full_vectors
    )

    result.update({
        "shap_top10_per_client": client_top10,
        "shap_top5_per_client": client_top5,
        "shap_full_vectors_per_client": client_full_vectors,
        "jaccard_pairwise": jaccard_pairwise,
        "divergence_full": divergence_full,
        "shared_features": shared_features,
        # FedPer yields one model per client: the divergence keys above
        # compare the three *personalized* assembled models, not one shared
        # global model evaluated per protocol (as under FedAvg/FedProx).
        "divergence_scope": "across personalized models (per-client assembled trunk+head)",
    })
    return result


def compare_explanations(
    fedper_run: dict,
    baseline_run: dict,
    baseline_label: str,
    run_label: str = "fedper",
) -> dict[str, dict]:
    """Compare each client's explanation from one run against the SAME client's
    model from another run at the same seed (e.g. FedAvg or local-only).

    Generic over the method being compared (FedPer, Cluster-FL, ...): the
    run's own top-5 list is stored under f"{run_label}_top5". For each
    protocol: top-5 Jaccard plus Spearman / cosine over the full 45-element
    mean-|SHAP| vectors.
    """
    comparisons: dict[str, dict] = {}
    for proto in PROTOCOLS:
        fedper_top5 = fedper_run["shap_top5_per_client"].get(proto, [])
        baseline_top5 = baseline_run["shap_top5_per_client"].get(proto, [])
        if fedper_top5 and baseline_top5:
            shared = sorted(set(fedper_top5).intersection(baseline_top5))
            top5_jaccard = round(len(shared) / len(set(fedper_top5).union(baseline_top5)), 4)
        else:
            shared, top5_jaccard = [], 0.0

        fedper_vec = fedper_run.get("shap_full_vectors_per_client", {}).get(proto)
        baseline_vec = baseline_run.get("shap_full_vectors_per_client", {}).get(proto)
        spearman = cosine = None
        if fedper_vec and baseline_vec:
            names_f = fedper_vec["feature_names"]
            names_b = baseline_vec["feature_names"]
            common = [n for n in names_f if n in set(names_b)]
            a = [fedper_vec["values"][names_f.index(n)] for n in common]
            b = [baseline_vec["values"][names_b.index(n)] for n in common]
            spearman, cosine = _spearman_cosine(a, b)
            spearman = round(spearman, 4) if spearman is not None else None
            cosine = round(cosine, 4) if cosine is not None else None

        comparisons[proto] = {
            f"{run_label}_top5": fedper_top5,
            f"{baseline_label}_top5": baseline_top5,
            "shared_top5": shared,
            "top5_jaccard": top5_jaccard,
            "spearman": spearman,
            "cosine": cosine,
        }
    return comparisons


def _to_runs_entry(run: dict) -> dict:
    """Project a train_fedper() result onto the study runs-array schema.

    Mirrors the keys stored for FedAvg/FedProx runs (aggregator/mu,
    client_metrics, SHAP keys, divergence keys) plus the FedPer-specific
    comparison sets; drops the training-only extras (history, model_paths).
    """
    keys = [
        "seed", "aggregator", "mu", "client_metrics", "avg_macro_f1",
        "shap_top10_per_client", "shap_top5_per_client",
        "shap_full_vectors_per_client", "jaccard_pairwise",
        "divergence_full", "shared_features", "divergence_scope",
        "fedper_vs_fedavg", "fedper_vs_local_only",
    ]
    return {k: run[k] for k in keys if k in run}


def run_fedper_suite(
    runs_data: list[dict],
    local_only_runs: list[dict] | None = None,
) -> list[dict]:
    """Run FedPer for both seeds and attach the per-seed comparison sets.

    fedper_vs_fedavg compares each client's FedPer explanation against the
    same client's FedAvg run at the same seed (looked up in runs_data).
    If local_only_runs is provided, fedper_vs_local_only is computed the
    same way against the local-only control at the same seed.
    """
    fedper_runs: list[dict] = []
    for seed in SEEDS:
        print(f"\n>>> Running FedPer (seed={seed})...")
        t0 = time.time()
        run = train_fedper(seed=seed)
        print(f">>> FedPer (seed={seed}) training + SHAP completed in "
              f"{time.time() - t0:.1f}s")

        fedavg_run = next(
            (r for r in runs_data if r["aggregator"] == "fedavg" and r["seed"] == seed), None
        )
        if fedavg_run is not None:
            run["fedper_vs_fedavg"] = compare_explanations(run, fedavg_run, "fedavg")

        if local_only_runs:
            local_run = next(
                (r for r in local_only_runs if r["seed"] == seed), None
            )
            if local_run is not None:
                run["fedper_vs_local_only"] = compare_explanations(run, local_run, "local_only")

        fedper_runs.append(_to_runs_entry(run))

    return fedper_runs


def main() -> None:
    """Standalone entry point: run FedPer for both seeds and merge into study_results.json.

    Re-run this only when a FedPer run needs regenerating; training is
    deterministic given the seed. Existing keys in study_results.json are
    left untouched; the 'runs' array entries with aggregator='fedper' are
    replaced.
    """
    if os.path.exists(STUDY_RESULTS_PATH):
        with open(STUDY_RESULTS_PATH, "r", encoding="utf-8") as f:
            study = json.load(f)
    else:
        study = {}

    runs_data = [r for r in study.get("runs", []) if r.get("aggregator") != "fedper"]
    local_only_runs = study.get("local_only", [])
    fedper_runs = run_fedper_suite(runs_data, local_only_runs)

    study["runs"] = runs_data + fedper_runs
    with open(STUDY_RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(study, f, indent=2)
    print(f"[OK] FedPer results merged into {STUDY_RESULTS_PATH} "
          f"(aggregator='fedper', seeds 42 and 7)")


if __name__ == "__main__":
    main()
