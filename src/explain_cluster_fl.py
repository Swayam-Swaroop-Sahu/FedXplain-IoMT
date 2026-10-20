"""SHAP explanations and cross-client divergence for Cluster-FL."""

import os
import json
import argparse
from pathlib import Path

os.environ["NUMBA_DISABLE_JIT"] = "1"

import numpy as np
import torch

from src.config import PROTOCOLS, DATA_DIR, SHAP_BACKGROUND_SIZE, SHAP_ATTACK_SAMPLE_SIZE
from src.model import IoMTMLP
from src.utils import get_feature_names, get_input_dim, set_seed
from src.explain import explain_client, compute_pairwise_divergence


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True, choices=[7, 42])
    parser.add_argument("--data-dir", default=DATA_DIR)
    parser.add_argument("--background-size", type=int, default=SHAP_BACKGROUND_SIZE)
    parser.add_argument("--attack-size", type=int, default=SHAP_ATTACK_SAMPLE_SIZE)
    args = parser.parse_args()

    set_seed(args.seed)

    input_dim = get_input_dim("wifi", data_dir=args.data_dir)
    feature_names = get_feature_names("wifi", data_dir=args.data_dir)

    checkpoint_dir = Path("models")
    result_dir = Path("results")
    result_dir.mkdir(parents=True, exist_ok=True)

    with open(result_dir / f"cluster_fl_seed{args.seed}.json") as f:
        cluster_results = json.load(f)

    clusters = cluster_results["clusters"]
    client_cluster = cluster_results["client_cluster"]

    client_top10 = {}
    client_top5 = {}
    client_vectors = {}

    for protocol in PROTOCOLS:
        cluster_id = str(client_cluster[protocol])
        checkpoint = checkpoint_dir / (
            f"cluster_fl_seed{args.seed}_cluster{cluster_id}.pth"
        )

        if not checkpoint.exists():
            raise FileNotFoundError(f"Missing checkpoint: {checkpoint}")

        model = IoMTMLP(input_dim)
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        model.eval()

        print(
            f"\nSeed {args.seed} | Client {protocol} | "
            f"Cluster {cluster_id} | Checkpoint {checkpoint}"
        )

        top10, top5, full_vector = explain_client(
            model=model,
            proto=protocol,
            feature_names=feature_names,
            data_dir=args.data_dir,
            seed=args.seed,
            bg_size=args.background_size,
            attack_sample_size=args.attack_size,
        )

        client_top10[protocol] = [
            {"feature": name, "mean_abs_shap": value}
            for name, value in top10
        ]
        client_top5[protocol] = top5
        client_vectors[protocol] = full_vector

    jaccard, shared_features, divergence = compute_pairwise_divergence(
        client_top5, client_vectors
    )

    output = {
        "method": "Cluster-FL",
        "seed": args.seed,
        "clusters": clusters,
        "client_cluster": client_cluster,
        "config": {
            "data_dir": args.data_dir,
            "background_size": args.background_size,
            "attack_sample_size": args.attack_size,
        },
        "client_top10": client_top10,
        "client_top5": client_top5,
        "shap_full_vectors_per_client": client_vectors,
        "jaccard_pairwise": jaccard,
        "shared_features": shared_features,
        "divergence_full": divergence,
    }

    output_path = result_dir / f"cluster_fl_shap_seed{args.seed}.json"
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print("\nTop-5 Jaccard:")
    print(json.dumps(jaccard, indent=2))

    print("\nSpearman and cosine similarity:")
    print(json.dumps(divergence, indent=2))

    print(f"\nSaved SHAP results to {output_path}")


if __name__ == "__main__":
    main()
