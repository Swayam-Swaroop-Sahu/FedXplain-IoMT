"""Cluster-FL experiment for FedXplain-IoMT.

From the repository root:
  PYTHONPATH=. python src/run_cluster_fl.py             # seed 7
  PYTHONPATH=. python src/run_cluster_fl.py --seed 42   # seed 42
"""
import argparse
import itertools
import json
from pathlib import Path

import numpy as np

from src.client import IoMTFlowerClient
from src.run_federated import fedavg_weighted
from src.utils import set_seed

import torch

CLIENTS = ("wifi", "mqtt", "bluetooth")


def flatten(params):
    return np.concatenate([np.asarray(p, dtype=np.float64).ravel() for p in params])


def cosine(a, b):
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 1.0 if np.linalg.norm(a) == 0 and np.linalg.norm(b) == 0 else 0.0
    return float(np.dot(a, b) / denom)


def choose_partition(sim):
    # Enumerate unique two-way partitions; Wi-Fi is fixed on the left to
    # avoid considering A|B and B|A as separate partitions.
    candidates = []
    remaining = CLIENTS[1:]
    for mask in range(1, 2 ** len(remaining)):
        left, right = [CLIENTS[0]], []
        for i, client in enumerate(remaining):
            (right if mask & (1 << i) else left).append(client)
        if not right:
            continue
        cross = [sim[a][b] for a in left for b in right]
        candidates.append({
            "clusters": [left, right],
            "max_cross_similarity": float(max(cross)),
            "mean_cross_similarity": float(np.mean(cross)),
        })
    candidates.sort(key=lambda x: (
        x["max_cross_similarity"], x["mean_cross_similarity"],
        tuple(x["clusters"][0]), tuple(x["clusters"][1])
    ))
    return candidates[0]["clusters"], candidates


def fit_client(client, params, epochs):
    # Flower NumPyClient.fit returns (parameters, number_of_examples, metrics).
    fitted, n, metrics = client.fit(
        [np.asarray(p).copy() for p in params], {"epochs": epochs}
    )
    return [np.asarray(p).copy() for p in fitted], int(n), dict(metrics or {})


def evaluate_client(client, params):
    loss, n, metrics = client.evaluate(
        [np.asarray(p).copy() for p in params], {}
    )
    return float(loss), int(n), dict(metrics or {})


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--warmup-rounds", type=int, default=5)
    parser.add_argument("--cluster-rounds", type=int, default=5)
    parser.add_argument("--local-epochs", type=int, default=3)
    parser.add_argument("--data-dir", default="data/study")
    parser.add_argument("--output", default=None)
    parser.add_argument("--models-dir", default="models")
    args = parser.parse_args(argv)

    if min(args.warmup_rounds, args.cluster_rounds, args.local_epochs) < 1:
        raise ValueError("Round counts and local epochs must be >= 1.")

    set_seed(args.seed)
    print(f"Cluster-FL | seed={args.seed} | warm-up={args.warmup_rounds} | "
          f"cluster rounds={args.cluster_rounds} | local epochs={args.local_epochs}")

    clients = {
        protocol: IoMTFlowerClient(
            cid=str(i), protocol=protocol, data_dir=args.data_dir, seed=args.seed
        )
        for i, protocol in enumerate(CLIENTS)
    }

    global_params = [np.asarray(p).copy() for p in clients["wifi"].get_parameters({})]
    warmup_history = []
    final_updates = {}

    # Phase 1: global FedAvg warm-up.
    for rnd in range(1, args.warmup_rounds + 1):
        print(f"\\n[Warm-up {rnd}/{args.warmup_rounds}]")
        fit_results, round_updates = [], {}
        for protocol in CLIENTS:
            local_params, n_train, _ = fit_client(
                clients[protocol], global_params, args.local_epochs
            )
            delta = flatten([local - base for local, base in zip(local_params, global_params)])
            round_updates[protocol] = delta
            fit_results.append((local_params, n_train))
            print(f"  {protocol:10s} n_train={n_train:6d} update_norm={np.linalg.norm(delta):.6f}")
        global_params = [np.asarray(p).copy() for p in fedavg_weighted(fit_results)]
        final_updates = round_updates
        warmup_history.append({
            "round": rnd,
            "update_norms": {c: float(np.linalg.norm(round_updates[c])) for c in CLIENTS}
        })

    # Phase 2: cluster based on cosine similarities of final warm-up updates.
    sim = {a: {b: cosine(final_updates[a], final_updates[b]) for b in CLIENTS}
           for a in CLIENTS}
    print("\\nFinal warm-up update cosine similarities:")
    for a in CLIENTS:
        print("  " + a + ": " + ", ".join(f"{b}={sim[a][b]:+.4f}" for b in CLIENTS))

    clusters, candidates = choose_partition(sim)
    print("\\nCandidate partitions:")
    for item in candidates:
        print(f"  {item['clusters'][0]} | {item['clusters'][1]} | "
              f"max_cross_sim={item['max_cross_similarity']:+.4f}, "
              f"mean_cross_sim={item['mean_cross_similarity']:+.4f}")
    print("\\nSelected clusters:")
    for i, members in enumerate(clusters):
        print(f"  Cluster {i}: {members}")

    # Phase 3: train one model per cluster, initialized from the warm-up model.
    cluster_params = {
        str(i): [np.asarray(p).copy() for p in global_params]
        for i in range(len(clusters))
    }
    cluster_history = []
    for rnd in range(1, args.cluster_rounds + 1):
        print(f"\\n[Cluster round {rnd}/{args.cluster_rounds}]")
        round_info = {"round": rnd, "clusters": {}}
        for i, members in enumerate(clusters):
            key = str(i)
            fit_results, per_client = [], {}
            for protocol in members:
                local_params, n_train, fit_metrics = fit_client(
                    clients[protocol], cluster_params[key], args.local_epochs
                )
                fit_results.append((local_params, n_train))
                per_client[protocol] = {"n_train": n_train, "fit_metrics": fit_metrics}
            cluster_params[key] = [
                np.asarray(p).copy() for p in fedavg_weighted(fit_results)
            ]
            round_info["clusters"][key] = {"clients": members, "client_fit": per_client}
            print(f"  Cluster {i} {members}: aggregated {len(fit_results)} client(s)")
        cluster_history.append(round_info)

    # Phase 4: evaluate each client against its assigned cluster model.
    evaluation, client_cluster = {}, {}
    print("\\nFinal client evaluation:")
    for i, members in enumerate(clusters):
        key = str(i)
        evaluation[key] = {}
        for protocol in members:
            client_cluster[protocol] = i
            loss, n_test, metrics = evaluate_client(clients[protocol], cluster_params[key])
            evaluation[key][protocol] = {"loss": loss, "n_test": n_test, **metrics}
            print(f"  {protocol:10s} n_test={n_test:6d} "
                  f"accuracy={metrics.get('accuracy', float('nan')):.4f} "
                  f"binary_F1={metrics.get('f1_binary', float('nan')):.4f} "
                  f"macro_F1={metrics.get('f1_macro', float('nan')):.4f}")

    # Save final model checkpoints for later SHAP analysis.
    from src.model import IoMTMLP
    from src.utils import get_input_dim

    models_dir = Path(args.models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    input_dim = get_input_dim("wifi", data_dir=args.data_dir)

    for cluster_id, params in cluster_params.items():
        model = IoMTMLP(input_dim)

        # Load the trained NumPy parameters into the PyTorch model.
        state_dict = model.state_dict()

        if len(params) != len(state_dict):
            raise ValueError(
                f"Parameter mismatch for cluster {cluster_id}: "
                f"{len(params)} arrays versus {len(state_dict)} state entries."
            )

        for (name, tensor), array in zip(state_dict.items(), params):
            array_tensor = torch.as_tensor(array, dtype=tensor.dtype)

            if array_tensor.shape != tensor.shape:
                raise ValueError(
                    f"Shape mismatch for {name}: "
                    f"checkpoint {tuple(array_tensor.shape)} versus "
                    f"model {tuple(tensor.shape)}"
                )

            state_dict[name] = array_tensor

        model.load_state_dict(state_dict)
        model.eval()

        checkpoint_path = (
            models_dir / f"cluster_fl_seed{args.seed}_cluster{cluster_id}.pth"
        )

        torch.save(model.state_dict(), checkpoint_path)
        print(f"Saved model checkpoint: {checkpoint_path}")
    output_path = Path(args.output or f"results/cluster_fl_seed{args.seed}.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "method": "Cluster-FL",
        "seed": args.seed,
        "config": {
            "warmup_rounds": args.warmup_rounds,
            "cluster_rounds": args.cluster_rounds,
            "total_rounds": args.warmup_rounds + args.cluster_rounds,
            "local_epochs": args.local_epochs,
            "data_dir": args.data_dir,
            "clients": list(CLIENTS),
            "clustering_method": "two-way partition minimizing maximum cross-cluster cosine similarity of final warm-up local updates",
        },
        "clusters": {str(i): members for i, members in enumerate(clusters)},
        "client_cluster": client_cluster,
        "update_cosine_similarity": sim,
        "candidate_partitions": candidates,
        "warmup_history": warmup_history,
        "cluster_history": cluster_history,
        "evaluation": evaluation,
    }
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, default=lambda x: x.item() if isinstance(x, np.generic) else str(x))
    print(f"\\nSaved results to {output_path.resolve()}")


if __name__ == "__main__":
    main()
