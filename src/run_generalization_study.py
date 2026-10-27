"""Parallel experiment script: Zero-Day threat transfer and Small-Clinic sample efficiency.

Writes ONLY results/generalization_results.json. It never modifies
results/study_results.json (read-only, used for sanity cross-checks) or any
existing checkpoint: models are loaded from models/ when present, otherwise
retrained deterministically (same seeds/config as the study) into
models/generalization/.

Run from the repo root:
    python -m src.run_generalization_study
"""

import json
import math
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.client import IoMTFlowerClient
from src.config import (
    BATCH_SIZE,
    DATA_DIR,
    EXPECTED_FEATURE_COUNT,
    FEDPER_SHARED_PARAM_COUNT,
    LEARNING_RATE,
    MODELS_DIR,
    N_LOCAL_EPOCHS,
    N_ROUNDS,
    PROTOCOLS,
    RESULTS_DIR,
    SEEDS,
    STUDY_RESULTS_PATH,
    TEST_SIZE,
)
from src.model import IoMTMLP
from src.run_cluster_fl import main as run_cluster_fl_main
from src.run_federated import fedavg_weighted, train_fedavg
from src.run_fedper import merge_weights, split_weights, train_fedper
from src.utils import _find_csv_path, _is_drop_column, load_and_preprocess, set_seed

GENERALIZATION_RESULTS_PATH = os.path.join(RESULTS_DIR, "generalization_results.json")
GEN_MODELS_DIR = os.path.join(MODELS_DIR, "generalization")

ZERO_DAY_TARGETS = ["bluetooth", "mqtt"]
SCARCITY_TARGETS = ["bluetooth", "mqtt", "wifi"]
SAMPLE_SIZES = [50, 100, 250, 500, 1000, 5000]
# Same optimiser budget for both arms. The 33-param head needs thousands of Adam
# steps to converge (200 steps left it under-trained; 2000 vs 5000 differ by <0.012 F1).
SCARCITY_EPOCHS = 10
SCARCITY_MIN_STEPS = 5000
SCARCITY_REPEATS = 3
# Matches run_local_only.py (N_ROUNDS x N_LOCAL_EPOCHS) so checkpoints are comparable.
LOCAL_ONLY_EPOCHS = N_ROUNDS * N_LOCAL_EPOCHS
METHODS = ["local_only", "fedavg", "fedper", "cluster_fl"]


# ----------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------
_RAW_CACHE: dict[tuple, tuple] = {}


def load_raw_split(protocol: str, seed: int, data_dir: str = DATA_DIR) -> tuple:
    """Unscaled features + string labels, using the exact split of load_and_preprocess().

    Returns (X_train, X_test, y_train, y_test, labels_train, labels_test) as NumPy arrays.
    """
    key = (protocol, seed, data_dir)
    if key in _RAW_CACHE:
        return _RAW_CACHE[key]

    df = pd.read_csv(_find_csv_path(protocol, data_dir))
    label_col = next(c for c in df.columns if c.strip().lower() in {"label", "labels"})
    labels = df[label_col].astype(str).str.strip().to_numpy()
    y = (np.char.lower(labels.astype(str)) != "benign").astype(int)

    X = df.drop(columns=[c for c in df.columns if _is_drop_column(c)], errors="ignore")
    X = X.drop(columns=X.select_dtypes(exclude=[np.number]).columns.tolist())
    X = np.nan_to_num(X.values.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    assert X.shape[1] == EXPECTED_FEATURE_COUNT, f"{protocol}: {X.shape[1]} features"

    split = train_test_split(X, y, labels, test_size=TEST_SIZE, stratify=y, random_state=seed)

    # Guardrail: the split + scaling must reproduce the study pipeline bit-for-bit.
    _, X_test_ref, _, _ = load_and_preprocess(protocol, data_dir=data_dir, random_state=seed)
    X_test_ours = scale(StandardScaler().fit(split[0]), split[1])
    assert torch.allclose(X_test_ours, X_test_ref, atol=1e-5), (
        f"{protocol} seed={seed}: split/scaling diverged from load_and_preprocess()"
    )

    _RAW_CACHE[key] = tuple(split)
    return _RAW_CACHE[key]


def scale(scaler: StandardScaler, X: np.ndarray) -> torch.Tensor:
    X_s = scaler.transform(X).astype(np.float32)
    return torch.from_numpy(np.nan_to_num(X_s, nan=0.0, posinf=0.0, neginf=0.0))


# ----------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------
@torch.no_grad()
def predict(model: nn.Module, X: torch.Tensor) -> np.ndarray:
    model.eval()
    return (model(X) >= 0.5).numpy().ravel().astype(int)


def binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    benign, attack = y_true == 0, y_true == 1
    return {
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "benign_recall": float(np.mean(y_pred[benign] == 0)) if benign.any() else None,
        "attack_recall": float(np.mean(y_pred[attack] == 1)) if attack.any() else None,
        "n_benign": int(benign.sum()),
        "n_attack": int(attack.sum()),
    }


def _mean(vals: list) -> float | None:
    clean = [v for v in vals if v is not None]
    return float(np.mean(clean)) if clean else None


def _std(vals: list) -> float | None:
    clean = [v for v in vals if v is not None]
    return float(np.std(clean)) if clean else None


# ----------------------------------------------------------------------
# Models: load existing checkpoint if present, otherwise retrain into GEN_MODELS_DIR
# ----------------------------------------------------------------------
def _load_state(path: str) -> IoMTMLP:
    model = IoMTMLP(EXPECTED_FEATURE_COUNT)
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    model.eval()
    return model


def _first_existing(paths: list[str]) -> str | None:
    return next((p for p in paths if os.path.exists(p)), None)


_LOCAL_CACHE: dict[tuple, IoMTMLP] = {}


def get_local_only_model(protocol: str, seed: int) -> tuple[IoMTMLP, str]:
    """Replicates run_local_only.train_local_only() training (same seeding, epochs) without SHAP."""
    key = (protocol, seed)
    if key not in _LOCAL_CACHE:
        set_seed(seed)
        client = IoMTFlowerClient(cid=0, protocol=protocol, seed=seed)
        weights, _, _ = client.fit(client.get_parameters(), {"epochs": LOCAL_ONLY_EPOCHS})
        model = IoMTMLP(EXPECTED_FEATURE_COUNT)
        model.set_weights(weights)
        model.eval()
        _LOCAL_CACHE[key] = model
    return _LOCAL_CACHE[key], f"trained in-process ({LOCAL_ONLY_EPOCHS} epochs)"


def get_fedavg_model(seed: int) -> tuple[IoMTMLP, str]:
    gen_path = os.path.join(GEN_MODELS_DIR, f"fedavg_seed{seed}.pth")
    path = _first_existing([
        os.path.join(MODELS_DIR, f"study_fedavg_seed{seed}.pth"),
        os.path.join(MODELS_DIR, f"fedavg_seed{seed}.pth"),
        gen_path,
    ])
    if path is None:
        os.makedirs(GEN_MODELS_DIR, exist_ok=True)
        train_fedavg(seed=seed, save_path=gen_path)
        path = gen_path
    return _load_state(path), path


def get_fedper_model(protocol: str, seed: int) -> tuple[IoMTMLP, str]:
    name = f"study_fedper_seed{seed}_{protocol}.pth"
    path = _first_existing([os.path.join(MODELS_DIR, name), os.path.join(GEN_MODELS_DIR, name)])
    if path is None:
        os.makedirs(GEN_MODELS_DIR, exist_ok=True)
        train_fedper(seed=seed, save_dir=GEN_MODELS_DIR, compute_shap=False)
        path = os.path.join(GEN_MODELS_DIR, name)
    return _load_state(path), path


def _study_cluster_assignment(seed: int) -> dict[str, int]:
    if not os.path.exists(STUDY_RESULTS_PATH):
        return {}
    with open(STUDY_RESULTS_PATH, "r", encoding="utf-8") as f:
        runs = json.load(f).get("runs", [])
    run = next((r for r in runs if r.get("aggregator") == "cluster_fl" and r.get("seed") == seed), None)
    return dict(run.get("clusters", {})) if run else {}


def get_cluster_fl_model(protocol: str, seed: int) -> tuple[IoMTMLP, str]:
    cid = _study_cluster_assignment(seed).get(protocol)
    if cid is not None:
        path = os.path.join(MODELS_DIR, f"cluster_fl_seed{seed}_cluster{cid}.pth")
        if os.path.exists(path):
            return _load_state(path), path

    # Retrained clustering may assign different ids, so use its own mapping.
    meta_path = os.path.join(GEN_MODELS_DIR, f"cluster_fl_seed{seed}.json")
    if not os.path.exists(meta_path):
        os.makedirs(GEN_MODELS_DIR, exist_ok=True)
        run_cluster_fl_main([
            "--seed", str(seed), "--data-dir", DATA_DIR,
            "--models-dir", GEN_MODELS_DIR, "--output", meta_path,
        ])
    with open(meta_path, "r", encoding="utf-8") as f:
        cid = json.load(f)["client_cluster"][protocol]
    path = os.path.join(GEN_MODELS_DIR, f"cluster_fl_seed{seed}_cluster{cid}.pth")
    return _load_state(path), path


def get_model(method: str, protocol: str, seed: int) -> tuple[IoMTMLP, str]:
    if method == "local_only":
        return get_local_only_model(protocol, seed)
    if method == "fedavg":
        return get_fedavg_model(seed)
    if method == "fedper":
        return get_fedper_model(protocol, seed)
    if method == "cluster_fl":
        return get_cluster_fl_model(protocol, seed)
    raise ValueError(method)


def get_fedper_trunk_excluding(target: str, seed: int) -> tuple[list[np.ndarray], str]:
    """FedPer trunk federated WITHOUT the target client (late-joining clinic).

    Same loop as run_fedper.train_fedper() restricted to the remaining clients.
    """
    path = os.path.join(GEN_MODELS_DIR, f"fedper_trunk_excl_{target}_seed{seed}.pth")
    if os.path.exists(path):
        return _load_state(path).get_weights()[:FEDPER_SHARED_PARAM_COUNT], path

    protos = [p for p in PROTOCOLS if p != target]
    print(f"\n>>> Training FedPer trunk without '{target}' (clients={protos}, seed={seed})")
    set_seed(seed)
    init = IoMTMLP(EXPECTED_FEATURE_COUNT).get_weights()
    trunk = init[:FEDPER_SHARED_PARAM_COUNT]
    heads = {p: [w.copy() for w in init[FEDPER_SHARED_PARAM_COUNT:]] for p in protos}
    for _ in range(N_ROUNDS):
        fit_results = []
        for cid, proto in enumerate(protos):
            client = IoMTFlowerClient(cid=cid, protocol=proto, seed=seed)
            weights, n_train, _ = client.fit(merge_weights(trunk, heads[proto]), {"epochs": N_LOCAL_EPOCHS})
            new_trunk, heads[proto] = split_weights(weights)
            fit_results.append((new_trunk, n_train))
        trunk = fedavg_weighted(fit_results)

    os.makedirs(GEN_MODELS_DIR, exist_ok=True)
    holder = IoMTMLP(EXPECTED_FEATURE_COUNT)
    holder.set_weights(merge_weights(trunk, init[FEDPER_SHARED_PARAM_COUNT:]))
    torch.save(holder.state_dict(), path)
    return trunk, path


def _reported_metric(study: dict, method: str, seed: int, protocol: str, key: str) -> float | None:
    if method == "local_only":
        run = next((r for r in study.get("local_only", []) if r.get("seed") == seed), None)
    else:
        run = next((r for r in study.get("runs", [])
                    if r.get("aggregator") == method and r.get("seed") == seed), None)
    return run.get("client_metrics", {}).get(protocol, {}).get(key) if run else None


# ----------------------------------------------------------------------
# Experiment 1: Zero-Day / novel attack generalization
# ----------------------------------------------------------------------
def run_zero_day_experiment(seeds: list[int] = SEEDS, targets: list[str] = ZERO_DAY_TARGETS) -> dict:
    """Evaluate each target client's model on attack types absent from its training set.

    Unseen attacks are drawn from the OTHER protocols' held-out test splits (same
    seed), so FL models never saw these exact flows. All inputs are standardized
    with the TARGET client's train-fitted scaler (what the deployed device would
    do). Foreign benign flows are scored the same way as a control: a model that
    flags everything foreign is detecting distribution shift, not the attack.
    """
    study = {}
    if os.path.exists(STUDY_RESULTS_PATH):
        with open(STUDY_RESULTS_PATH, "r", encoding="utf-8") as f:
            study = json.load(f)

    out: dict = {
        "config": {
            "targets": targets,
            "seeds": seeds,
            "methods": METHODS,
            "scaler": "target client's StandardScaler fitted on its own train split",
            "unseen_source": "held-out test split of the other protocols (same seed)",
            "local_only_epochs": LOCAL_ONLY_EPOCHS,
        },
        "targets": {},
    }

    for target in targets:
        print(f"\n===== Zero-day: target={target} =====")
        per_seed = []
        seen_labels: set[str] = set()
        unseen_meta: dict[str, dict] = {}
        for seed in seeds:
            X_tr, X_te, _, y_te, l_tr, _ = load_raw_split(target, seed)
            seen_here = set(l_tr.tolist())
            seen_labels |= seen_here
            scaler = StandardScaler().fit(X_tr)
            X_in = scale(scaler, X_te)

            models, checkpoints = {}, {}
            for m in METHODS:
                models[m], checkpoints[m] = get_model(m, target, seed)

            entry: dict = {
                "seed": seed,
                "checkpoints": checkpoints,
                "in_distribution": {},
                "unseen_attack_recall": {m: {} for m in METHODS},
                "foreign_benign_flag_rate": {m: {} for m in METHODS},
            }
            for m in METHODS:
                metrics = binary_metrics(y_te, predict(models[m], X_in))
                metrics["reported_f1_macro"] = _reported_metric(study, m, seed, target, "f1_macro")
                metrics["reported_benign_recall"] = _reported_metric(study, m, seed, target, "benign_recall")
                entry["in_distribution"][m] = metrics

            for source in PROTOCOLS:
                if source == target:
                    continue
                _, X_src, _, _, _, l_src = load_raw_split(source, seed)
                X_src_t = scale(scaler, X_src)
                preds = {m: predict(models[m], X_src_t) for m in METHODS}
                for label in sorted(set(l_src.tolist())):
                    mask = l_src == label
                    if label.lower() == "benign":
                        for m in METHODS:
                            entry["foreign_benign_flag_rate"][m][source] = float(np.mean(preds[m][mask] == 1))
                    elif label not in seen_here:
                        unseen_meta.setdefault(label, {"source": source, "n_test": {}})
                        unseen_meta[label]["n_test"][str(seed)] = int(mask.sum())
                        for m in METHODS:
                            entry["unseen_attack_recall"][m][label] = float(np.mean(preds[m][mask] == 1))
            per_seed.append(entry)

            for m in METHODS:
                rec = entry["unseen_attack_recall"][m]
                print(f"  seed={seed} {m:<11s} in-dist F1={entry['in_distribution'][m]['f1_macro']:.4f} "
                      f"| mean unseen-attack recall={np.mean(list(rec.values())):.4f} "
                      f"| foreign-benign flag={entry['foreign_benign_flag_rate'][m]}")

        unseen_labels = sorted(l for l in unseen_meta if l not in seen_labels)
        summary: dict = {}
        for m in METHODS:
            per_attack = {}
            for label in unseen_labels:
                vals = {str(e["seed"]): e["unseen_attack_recall"][m].get(label) for e in per_seed}
                per_attack[label] = {
                    "source": unseen_meta[label]["source"],
                    "mean": _mean(list(vals.values())),
                    "per_seed": vals,
                }
            fb = {src: _mean([e["foreign_benign_flag_rate"][m].get(src) for e in per_seed])
                  for src in PROTOCOLS if src != target}
            # Margin > 0 means attacks are flagged more often than benign traffic from the same protocol.
            margins = [per_attack[l]["mean"] - fb[per_attack[l]["source"]] for l in unseen_labels
                       if per_attack[l]["mean"] is not None and fb.get(per_attack[l]["source"]) is not None]
            ind = [e["in_distribution"][m] for e in per_seed]
            summary[m] = {
                "per_attack": per_attack,
                "mean_unseen_attack_recall": _mean([v["mean"] for v in per_attack.values()]),
                "foreign_benign_flag_rate": fb,
                "mean_detection_margin": _mean(margins),
                "in_distribution": {
                    k: _mean([x[k] for x in ind]) for k in ("f1_macro", "benign_recall", "attack_recall")
                },
            }

        out["targets"][target] = {
            "seen_train_labels": sorted(seen_labels),
            "unseen_attacks": {l: unseen_meta[l] for l in unseen_labels},
            "summary": summary,
            "per_seed": per_seed,
        }
    return out


# ----------------------------------------------------------------------
# Experiment 2: Data scarcity / small-clinic sample efficiency
# ----------------------------------------------------------------------
def _train_params(model: IoMTMLP, params: list, X: torch.Tensor, y: torch.Tensor,
                  epochs: int, min_steps: int) -> int:
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X, y), batch_size=BATCH_SIZE, shuffle=True
    )
    n_epochs = max(epochs, math.ceil(min_steps / len(loader)))
    optimizer = torch.optim.Adam(params, lr=LEARNING_RATE)
    criterion = nn.BCELoss()
    model.train()
    for _ in range(n_epochs):
        for X_b, y_b in loader:
            optimizer.zero_grad()
            criterion(model(X_b), y_b).backward()
            optimizer.step()
    model.eval()
    return n_epochs * len(loader)


def _head_only_model(trunk: list[np.ndarray], init_seed: int) -> tuple[IoMTMLP, list]:
    set_seed(init_seed)
    model = IoMTMLP(EXPECTED_FEATURE_COUNT)
    model.set_weights(merge_weights(trunk, model.get_weights()[FEDPER_SHARED_PARAM_COUNT:]))
    head = model.net[5]
    for p in model.parameters():
        p.requires_grad = False
    for p in head.parameters():
        p.requires_grad = True
    params = list(head.parameters())
    assert sum(p.numel() for p in params) == 33
    return model, params


def run_data_scarcity_experiment(
    sample_sizes: list[int] = SAMPLE_SIZES,
    seeds: list[int] = SEEDS,
    target: str = "bluetooth",
    repeats: int = SCARCITY_REPEATS,
) -> dict:
    """Macro F1 / benign recall vs. training-set size N on the target client's full test split.

    Arms (identical optimiser budget per N):
      local_only          - fresh IoMTMLP, all 5,057 params trained on N samples.
      fedper_head         - frozen trunk from the standard FedPer run, fresh 33-param
                            head. NOTE: target participated in that federation with its
                            full train set, so this is an optimistic upper bound.
      fedper_head_unseen  - frozen trunk federated WITHOUT the target client (late-
                            joining clinic): no target data reached the trunk.
    The scaler is refit on the N-sample subset (a small clinic cannot see more).
    """
    sizes = list(sample_sizes) + ["full"]
    arms = ["local_only", "fedper_head", "fedper_head_unseen"]
    runs: list[dict] = []
    trunk_paths: dict[str, dict] = {}

    for seed in seeds:
        X_tr, X_te, y_tr, y_te, _, _ = load_raw_split(target, seed)
        fedper_model, fedper_path = get_fedper_model(target, seed)
        trunks = {
            "fedper_head": fedper_model.get_weights()[:FEDPER_SHARED_PARAM_COUNT],
        }
        trunks["fedper_head_unseen"], unseen_path = get_fedper_trunk_excluding(target, seed)
        trunk_paths[str(seed)] = {"fedper_head": fedper_path, "fedper_head_unseen": unseen_path}

        for n in sizes:
            for rep in range(repeats):
                draw_seed = seed * 1000 + rep
                if n == "full":
                    idx = np.arange(len(y_tr))
                else:
                    idx, _ = train_test_split(np.arange(len(y_tr)), train_size=n,
                                              stratify=y_tr, random_state=draw_seed)
                scaler = StandardScaler().fit(X_tr[idx])
                X_sub = scale(scaler, X_tr[idx])
                y_sub = torch.from_numpy(y_tr[idx].astype(np.float32)).unsqueeze(1)
                X_test = scale(scaler, X_te)

                for arm in arms:
                    if arm == "local_only":
                        set_seed(draw_seed)
                        model = IoMTMLP(EXPECTED_FEATURE_COUNT)
                        params = list(model.parameters())
                    else:
                        model, params = _head_only_model(trunks[arm], draw_seed)
                    set_seed(draw_seed)
                    steps = _train_params(model, params, X_sub, y_sub, SCARCITY_EPOCHS, SCARCITY_MIN_STEPS)
                    metrics = binary_metrics(y_te, predict(model, X_test))
                    runs.append({
                        "seed": seed, "repeat": rep, "n": n, "n_train": int(len(idx)),
                        "method": arm, "trainable_params": int(sum(p.numel() for p in params)),
                        "optimizer_steps": steps, **metrics,
                    })
                print(f"  seed={seed} N={n!s:>5} rep={rep} | " + " | ".join(
                    f"{r['method']}: F1={r['f1_macro']:.4f}" for r in runs[-len(arms):]))

    summary: dict[str, list[dict]] = {}
    for arm in arms:
        summary[arm] = []
        for n in sizes:
            sel = [r for r in runs if r["method"] == arm and r["n"] == n]
            summary[arm].append({
                "n": n,
                "n_train": sel[0]["n_train"],
                "n_runs": len(sel),
                **{f"{k}_mean": _mean([r[k] for r in sel]) for k in ("f1_macro", "benign_recall", "attack_recall")},
                **{f"{k}_std": _std([r[k] for r in sel]) for k in ("f1_macro", "benign_recall", "attack_recall")},
                "f1_macro_min": float(min(r["f1_macro"] for r in sel)),
            })

    return {
        "config": {
            "target": target,
            "sample_sizes": sizes,
            "seeds": seeds,
            "repeats_per_seed": repeats,
            "epochs": SCARCITY_EPOCHS,
            "min_optimizer_steps": SCARCITY_MIN_STEPS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "subsampling": "stratified draw from the target train split (random_state = seed*1000 + repeat)",
            "scaler": "refit on the N-sample subset",
            "evaluation": "full held-out target test split",
            "trunk_checkpoints": trunk_paths,
        },
        "methods": {
            "local_only": "Fresh IoMTMLP, all 5,057 parameters trained on N samples",
            "fedper_head": "Frozen FedPer trunk (target took part with full data; optimistic bound) + 33-param head",
            "fedper_head_unseen": "Frozen FedPer trunk federated without the target client + 33-param head",
        },
        "summary": summary,
        "runs": runs,
    }


if __name__ == "__main__":
    set_seed(42)
    t0 = time.time()
    zero_day_results = run_zero_day_experiment()
    scarcity_results = {"targets": {t: run_data_scarcity_experiment(target=t) for t in SCARCITY_TARGETS}}
    output = {
        "metadata": {
            "generated": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
            "torch_version": torch.__version__,
            "data_dir": DATA_DIR,
            "runtime_seconds": round(time.time() - t0, 1),
        },
        "zero_day": zero_day_results,
        "data_scarcity": scarcity_results,
    }
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(GENERALIZATION_RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"Results successfully saved to {GENERALIZATION_RESULTS_PATH}")
