"""Local-only baseline (control): each protocol trains alone, no aggregation."""

import json
import os
import sys

import torch

# Allow running as `python src/run_local_only.py` from repo root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import (
    BATCH_SIZE,
    DATA_DIR,
    LEARNING_RATE,
    N_LOCAL_EPOCHS,
    N_ROUNDS,
    PROTOCOLS,
    RESULTS_DIR,
    SEEDS,
    SHAP_ATTACK_SAMPLE_SIZE,
    SHAP_BACKGROUND_SIZE,
    STUDY_RESULTS_PATH,
)
from src.client import IoMTFlowerClient
from src.explain import (
    compute_pairwise_divergence,
    explain_client,
)
from src.utils import get_feature_names, set_seed


def train_local_only(
    seed: int = 42,
    data_dir: str = DATA_DIR,
    n_rounds: int = N_ROUNDS,
    n_epochs: int = N_LOCAL_EPOCHS,
) -> dict:
    """Train one isolated model per protocol (no sharing) and explain each.

    Total local epochs (n_rounds * n_epochs) matches the federated runs so
    per-client training compute is comparable. Reuses IoMTFlowerClient's
    preprocessing and evaluate() so metrics are computed identically.

    Returns a dict with the same schema as one entry of the study runs array
    (minus aggregator/mu), including full SHAP vectors and divergence_full.
    """
    total_epochs = n_rounds * n_epochs
    print(f"Starting local-only baseline: {total_epochs} epochs per client "
          f"({n_rounds} rounds x {n_epochs} epochs), seed={seed}, data_dir={data_dir}\n")

    feature_names = get_feature_names("wifi", data_dir=data_dir)
    client_metrics: dict[str, dict] = {}
    client_top10: dict[str, list[tuple[str, float]]] = {}
    client_top5: dict[str, list[str]] = {}
    client_full_vectors: dict[str, dict[str, list]] = {}

    for proto in PROTOCOLS:
        set_seed(seed)
        client = IoMTFlowerClient(cid=0, protocol=proto, data_dir=data_dir, seed=seed)

        # ----- Local training (fresh model, no global weights) -----
        model = client.model
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        criterion = torch.nn.BCELoss()
        dataset = torch.utils.data.TensorDataset(client.X_train, client.y_train)
        loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

        final_loss = 0.0
        model.train()
        for epoch in range(total_epochs):
            epoch_loss = 0.0
            for X_batch, y_batch in loader:
                optimizer.zero_grad()
                preds = model(X_batch)
                loss = criterion(preds, y_batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * X_batch.size(0)
            final_loss = epoch_loss / len(dataset)
            if (epoch + 1) % 10 == 0 or epoch == total_epochs - 1:
                print(f"  {proto.upper():<9} local epoch {epoch + 1}/{total_epochs} | Loss: {final_loss:.4f}")

        # ----- Evaluate with the same metric code as the federated runs -----
        _, _, metrics = client.evaluate(model.get_weights(), {})
        client_metrics[proto] = metrics
        print(f"  {proto.upper():<9} done -> Macro F1: {metrics['f1_macro']:.4f} | "
              f"Binary F1: {metrics['f1_binary']:.4f}\n")

        # ----- SHAP on the local model -----
        top10, top5, full_vec = explain_client(
            model,
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

    return {
        "seed": seed,
        "client_metrics": client_metrics,
        "shap_top10_per_client": client_top10,
        "shap_top5_per_client": client_top5,
        "shap_full_vectors_per_client": client_full_vectors,
        "jaccard_pairwise": jaccard_pairwise,
        "divergence_full": divergence_full,
        "shared_features": shared_features,
        # Local-only yields one isolated model per client: the divergence keys
        # compare three *different* models, not one shared model per protocol.
        "divergence_scope": "across local-only models (one isolated model per client)",
    }


def compute_local_vs_global(
    local_run: dict,
    fedavg_run: dict,
) -> dict[str, dict]:
    """Compare each local-only client's explanation against the SAME client under FedAvg.

    Both runs must use the same seed. For each protocol: top-5 Jaccard plus
    Spearman / cosine over the full 45-element mean-|SHAP| vectors.
    """
    comparisons: dict[str, dict] = {}
    for proto in PROTOCOLS:
        local_top5 = local_run["shap_top5_per_client"].get(proto, [])
        global_top5 = fedavg_run["shap_top5_per_client"].get(proto, [])
        if local_top5 and global_top5:
            shared = sorted(set(local_top5).intersection(global_top5))
            top5_jaccard = round(len(shared) / len(set(local_top5).union(global_top5)), 4)
        else:
            shared, top5_jaccard = [], 0.0

        local_vec = local_run.get("shap_full_vectors_per_client", {}).get(proto)
        global_vec = fedavg_run.get("shap_full_vectors_per_client", {}).get(proto)
        spearman = cosine = None
        if local_vec and global_vec:
            from src.explain import _spearman_cosine

            names_l = local_vec["feature_names"]
            names_g = global_vec["feature_names"]
            common = [n for n in names_l if n in set(names_g)]
            a = [local_vec["values"][names_l.index(n)] for n in common]
            b = [global_vec["values"][names_g.index(n)] for n in common]
            spearman, cosine = _spearman_cosine(a, b)
            spearman = round(spearman, 4) if spearman is not None else None
            cosine = round(cosine, 4) if cosine is not None else None

        comparisons[proto] = {
            "local_top5": local_top5,
            "fedavg_top5": global_top5,
            "shared_top5": shared,
            "top5_jaccard": top5_jaccard,
            "spearman": spearman,
            "cosine": cosine,
        }
    return comparisons


def run_local_only_suite(runs_data: list[dict] | None = None) -> dict:
    """Run local-only for all configured seeds and build the payload for study_results.json.

    If runs_data (the study 'runs' array) is provided, the per-seed
    local_vs_global comparisons are computed against each seed's FedAvg
    entry; otherwise they are looked up from the existing study_results.json.
    """
    local_runs = [train_local_only(seed=seed) for seed in SEEDS]

    if runs_data is None and os.path.exists(STUDY_RESULTS_PATH):
        with open(STUDY_RESULTS_PATH, "r", encoding="utf-8") as f:
            runs_data = json.load(f).get("runs", [])

    local_vs_global: dict[str, dict] = {}
    local_by_seed = {r["seed"]: r for r in local_runs}
    if runs_data:
        for seed in SEEDS:
            fedavg_run = next(
                (r for r in runs_data if r["aggregator"] == "fedavg" and r["seed"] == seed), None
            )
            if fedavg_run is not None and seed in local_by_seed:
                local_vs_global[f"seed{seed}_local_vs_fedavg_seed{seed}"] = compute_local_vs_global(
                    local_by_seed[seed], fedavg_run
                )

    return {"local_only": local_runs, "local_vs_global": local_vs_global}


def main() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    payload = run_local_only_suite()

    # Merge into the existing study results without touching other keys
    if os.path.exists(STUDY_RESULTS_PATH):
        with open(STUDY_RESULTS_PATH, "r", encoding="utf-8") as f:
            study = json.load(f)
    else:
        study = {}
    study.update(payload)
    with open(STUDY_RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(study, f, indent=2)
    print(f"[OK] Local-only results written to {STUDY_RESULTS_PATH} (keys: local_only, local_vs_global)")


if __name__ == "__main__":
    main()
