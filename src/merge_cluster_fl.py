"""Merge teammate-produced Cluster-FL runs into study_results.json.

Empty-slot workflow: the teammate's Cluster-FL pipeline drops a JSON array of
run entries at CLUSTER_FL_RESULTS_PATH following CLUSTER_FL_RUN_SCHEMA (see
src/run_study_experiments.py). This tool validates each entry, backfills the
cluster_fl_vs_fedavg / cluster_fl_vs_local_only comparison sets from stored
SHAP vectors when absent, and merges the runs into study_results.json.

No other key in study_results.json is touched; existing aggregator='cluster_fl'
entries are replaced. No training is performed. Usage (repo root):

    python src/merge_cluster_fl.py
"""

import json
import os
import sys

# Allow running as `python src/merge_cluster_fl.py` from repo root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import STUDY_RESULTS_PATH
from src.run_study_experiments import (
    enrich_cluster_fl_comparisons,
    load_cluster_fl_runs,
)


def merge_cluster_fl() -> list[dict]:
    """Validate, enrich, and merge Cluster-FL runs into study_results.json."""
    cluster_fl_runs = load_cluster_fl_runs()
    if not cluster_fl_runs:
        print("[OK] Cluster-FL slot is empty - nothing to merge "
              "(in progress, port pending).")
        return []

    if os.path.exists(STUDY_RESULTS_PATH):
        with open(STUDY_RESULTS_PATH, "r", encoding="utf-8") as f:
            study = json.load(f)
    else:
        raise FileNotFoundError(
            f"{STUDY_RESULTS_PATH} not found - the main suite must run first "
            f"so Cluster-FL comparisons have FedAvg/local-only baselines."
        )

    runs_data = study.get("runs", [])
    local_only_runs = study.get("local_only", [])
    enrich_cluster_fl_comparisons(cluster_fl_runs, runs_data, local_only_runs)

    study["runs"] = (
        [r for r in runs_data if r.get("aggregator") != "cluster_fl"] + cluster_fl_runs
    )
    with open(STUDY_RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(study, f, indent=2)
    print(f"[OK] Merged {len(cluster_fl_runs)} Cluster-FL run(s) into {STUDY_RESULTS_PATH}.")
    return cluster_fl_runs


if __name__ == "__main__":
    merge_cluster_fl()
