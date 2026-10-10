"""Backfill genuinely missing result keys into study_results.json.

Completion-pass tool: re-trains ONLY what is missing (the centralized
baseline for seeds not yet run) and computes every remaining derived key -
per-seed local-vs-FedAvg comparisons, local-only divergence scopes, the
all-method noise floor, and the consolidated divergence summary - from
already-stored SHAP vectors. No federated training and no SHAP recomputation.
Idempotent: re-running detects what is already present and skips it.

Usage (repo root):
    python scripts/backfill_completions.py
"""

import json
import os
import sys
import time

# Allow running as `python scripts/backfill_completions.py` from repo root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.centralized import main as run_centralized
from src.config import DATA_DIR, SEEDS, STUDY_RESULTS_PATH
from src.run_local_only import compute_local_vs_global
from src.run_study_experiments import build_divergence_summary, compute_noise_floor

LOCAL_ONLY_SCOPE = "across local-only models (one isolated model per client)"


def _load_study() -> dict:
    if not os.path.exists(STUDY_RESULTS_PATH):
        raise FileNotFoundError(f"{STUDY_RESULTS_PATH} not found - run the suite first.")
    with open(STUDY_RESULTS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _backfill_centralized(study: dict, actions: list[str]) -> None:
    """Ensure a centralized baseline exists for every seed (trains only missing seeds)."""
    centralized_by_seed = dict(study.get("centralized_by_seed", {}))
    # The legacy top-level 'centralized' key is the seed-42 run.
    if "42" not in centralized_by_seed and "centralized" in study:
        centralized_by_seed["42"] = study["centralized"]
        actions.append("centralized_by_seed: seeded entry 42 from legacy 'centralized' key")

    for seed in SEEDS:
        if str(seed) in centralized_by_seed:
            continue
        print(f">>> Backfill: training centralized baseline (seed={seed})...")
        t0 = time.time()
        centralized_by_seed[str(seed)] = run_centralized(data_dir=DATA_DIR, seed=seed)
        actions.append(f"centralized_by_seed: trained missing seed {seed} "
                       f"({time.time() - t0:.1f}s)")

    study["centralized_by_seed"] = centralized_by_seed


def _backfill_local_vs_global(study: dict, actions: list[str]) -> None:
    """Compute per-seed local-only vs FedAvg comparisons from stored SHAP vectors."""
    local_vs_global = dict(study.get("local_vs_global", {}))
    runs_data = study.get("runs", [])
    local_by_seed = {r["seed"]: r for r in study.get("local_only", [])}

    for seed in SEEDS:
        key = f"seed{seed}_local_vs_fedavg_seed{seed}"
        if key in local_vs_global or seed not in local_by_seed:
            continue
        fedavg_run = next(
            (r for r in runs_data if r.get("aggregator") == "fedavg" and r.get("seed") == seed),
            None,
        )
        if fedavg_run is None:
            continue
        local_vs_global[key] = compute_local_vs_global(local_by_seed[seed], fedavg_run)
        actions.append(f"local_vs_global: computed {key} from stored vectors")

    study["local_vs_global"] = local_vs_global


def _backfill_local_scopes(study: dict, actions: list[str]) -> None:
    """Label local-only divergence keys as comparing different models."""
    for run in study.get("local_only", []):
        if "divergence_scope" not in run:
            run["divergence_scope"] = LOCAL_ONLY_SCOPE
            actions.append(f"local_only seed {run['seed']}: added divergence_scope")


def _recompute_noise_floor(study: dict, actions: list[str]) -> None:
    """Recompute the noise floor across ALL methods; report any drift for
    pre-existing labels (values must not change, only new keys are added)."""
    old = study.get("preliminary_noise_floor", {})
    new = compute_noise_floor(study.get("runs", []), study.get("local_only", []))

    drift = []
    for label, entry in old.items():
        if label not in new:
            drift.append(f"{label}: missing after recompute")
            continue
        for proto, d in entry.get("per_client_top5_stability", {}).items():
            if new[label]["per_client_top5_stability"].get(proto, {}).get("seed_to_seed_jaccard") != d.get("seed_to_seed_jaccard"):
                drift.append(f"{label}/{proto}: seed_to_seed_jaccard changed")
        for pair, d in entry.get("pairwise_divergence_spread", {}).items():
            if new[label]["pairwise_divergence_spread"].get(pair, {}).get("abs_diff") != d.get("abs_diff"):
                drift.append(f"{label}/{pair}: jaccard abs_diff changed")
    if drift:
        raise RuntimeError("Noise-floor recompute drifted from stored values: " + "; ".join(drift))

    added = sorted(set(new) - set(old))
    actions.append(f"preliminary_noise_floor: recomputed for ALL methods "
                   f"(new labels: {added or 'none'}; every method now carries full_metric_spread)")
    study["preliminary_noise_floor"] = new


def _recompute_divergence_summary(study: dict, actions: list[str]) -> None:
    """Build the consolidated divergence summary (all methods, all seeds,
    all three metrics, plus the seed-to-seed noise-floor reference)."""
    study["divergence_summary"] = build_divergence_summary(
        study.get("runs", []), study.get("local_only", [])
    )
    actions.append("divergence_summary: built consolidated per-method x seed x pair block "
                   "(cluster_fl placeholder: slot empty)")


def main() -> None:
    study = _load_study()
    actions: list[str] = []

    _backfill_centralized(study, actions)
    _backfill_local_vs_global(study, actions)
    _backfill_local_scopes(study, actions)
    _recompute_noise_floor(study, actions)
    _recompute_divergence_summary(study, actions)

    with open(STUDY_RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(study, f, indent=2)

    print(f"\n[OK] Backfill complete - {len(actions)} action(s):")
    for a in actions:
        print(f"  - {a}")
    print(f"[OK] Written to {STUDY_RESULTS_PATH}")


if __name__ == "__main__":
    main()
