"""Completeness audit for study_results.json.

Prints a run x required-key completeness matrix and fails loudly on any gap:
for BOTH seeds (42, 7) the file must contain centralized, FedAvg, FedProx x
every mu, local-only, and FedPer runs, each with complete client_metrics,
top-10/top-5 SHAP rankings, full 45-dim SHAP vectors, Jaccard, and
Spearman/cosine divergence - plus the comparison keys for local-only and
FedPer, the all-method noise floor, the consolidated divergence summary,
and the documented (empty) cluster_fl slot.

Usage (repo root):
    python scripts/audit_results.py
Exit code 1 if any check fails.
"""

import json
import os
import sys

# Allow running as `python scripts/audit_results.py` from repo root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import MU_VALUES, PROTOCOLS, SEEDS, STUDY_RESULTS_PATH  # noqa: E402
from src.run_study_experiments import CLUSTER_FL_RUN_SCHEMA, PAIR_LABELS  # noqa: E402

REQUIRED_METRIC_KEYS = (
    "f1_macro", "f1_binary", "accuracy",
    "benign_precision", "benign_recall", "attack_precision", "attack_recall",
    "tn", "fp", "fn", "tp",
)


def _check_client_metrics(run: dict, problems: list[str], label: str) -> bool:
    ok = True
    for proto in PROTOCOLS:
        m = run.get("client_metrics", {}).get(proto, {})
        missing = [k for k in REQUIRED_METRIC_KEYS if k not in m]
        if missing:
            problems.append(f"{label} client_metrics[{proto}] missing {missing}")
            ok = False
    return ok


def _check_shap_keys(run: dict, problems: list[str], label: str) -> bool:
    ok = True
    top10 = run.get("shap_top10_per_client", {})
    top5 = run.get("shap_top5_per_client", {})
    full = run.get("shap_full_vectors_per_client", {})
    for proto in PROTOCOLS:
        if len(top10.get(proto, [])) < 10:
            problems.append(f"{label} shap_top10[{proto}] incomplete")
            ok = False
        if len(top5.get(proto, [])) < 5:
            problems.append(f"{label} shap_top5[{proto}] incomplete")
            ok = False
        vec = full.get(proto, {})
        if len(vec.get("values", [])) != 45 or len(vec.get("feature_names", [])) != 45:
            problems.append(f"{label} shap_full_vectors[{proto}] not 45-dim")
            ok = False
    return ok


def _check_divergence_keys(run: dict, problems: list[str], label: str) -> bool:
    ok = True
    j = run.get("jaccard_pairwise", {})
    d = run.get("divergence_full", {})
    for pair in PAIR_LABELS:
        if pair not in j:
            problems.append(f"{label} jaccard_pairwise missing {pair}")
            ok = False
        dp = d.get(pair, {})
        if dp.get("spearman") is None or dp.get("cosine") is None:
            problems.append(f"{label} divergence_full[{pair}] missing spearman/cosine")
            ok = False
    return ok


def _cell(ok: bool | None) -> str:
    return " OK " if ok else ("GAP " if ok is False else " n/a")


def main() -> int:
    if not os.path.exists(STUDY_RESULTS_PATH):
        print(f"FAIL: {STUDY_RESULTS_PATH} not found.")
        return 1
    with open(STUDY_RESULTS_PATH, "r", encoding="utf-8") as f:
        study = json.load(f)

    problems: list[str] = []
    rows: list[tuple[str, bool | None, bool | None, bool | None, bool | None, bool | None]] = []

    def audit_run(
        run: dict | None,
        label: str,
        need_scope: bool | None = False,
        cmp_key: str | None = None,
        applic: tuple[bool, bool, bool, bool] = (True, True, True, True),
    ) -> None:
        """Audit one run; applic flags which columns apply (False = n/a)."""
        if run is None:
            problems.append(f"{label}: RUN MISSING")
            rows.append((label, False, False, False, False, False))
            return
        a_met, a_shap, a_div, a_scope = applic
        m = _check_client_metrics(run, problems, label) if a_met else None
        s = _check_shap_keys(run, problems, label) if a_shap else None
        d = _check_divergence_keys(run, problems, label) if a_div else None
        if a_scope is None:
            scope = None
        else:
            scope = (not need_scope) or ("divergence_scope" in run)
            if need_scope and not scope:
                problems.append(f"{label} missing divergence_scope")
        cmp_ok = None
        if cmp_key:
            keys = cmp_key if isinstance(cmp_key, (tuple, list)) else (cmp_key,)
            missing = [k for k in keys if k not in run]
            cmp_ok = not missing
            if missing:
                problems.append(f"{label} missing comparison key(s) {missing}")
        rows.append((label, m, s, d, scope, cmp_ok))

    # --- Federated / personalized runs (runs array) ---
    runs = study.get("runs", [])
    fedper_cmp_keys = ("fedper_vs_fedavg", "fedper_vs_local_only")
    for seed in SEEDS:
        for agg, mu, label in (
            [("fedavg", None, "fedavg")] +
            [("fedprox", m, f"fedprox_mu{m}") for m in MU_VALUES] +
            [("fedper", None, "fedper")] +
            [("cluster_fl", None, "cluster_fl (slot)")]
        ):
            run = next(
                (r for r in runs if r.get("aggregator") == agg
                 and r.get("mu") == mu and r.get("seed") == seed),
                None,
            )
            if agg == "cluster_fl":
                # Empty slot: allowed to be absent; if present it must be complete.
                if run is None:
                    continue
                audit_run(run, f"{label} seed{seed}", True, "cluster_fl_vs_fedavg")
                continue
            if agg == "fedper":
                audit_run(run, f"{label} seed{seed}", True, fedper_cmp_keys)
                continue
            audit_run(run, f"{label} seed{seed}", False, None)

    # --- Local-only runs (top-level array) ---
    local_runs = {r.get("seed"): r for r in study.get("local_only", [])}
    for seed in SEEDS:
        run = local_runs.get(seed)
        lvg_key = f"seed{seed}_local_vs_fedavg_seed{seed}"
        lvg_ok = lvg_key in study.get("local_vs_global", {})
        if run is not None and not lvg_ok:
            problems.append(f"local_only seed{seed}: missing local_vs_global['{lvg_key}']")
        audit_run(run, f"local_only seed{seed}", True, None)
        # Record the cross-file comparison result in the cmp column.
        if rows[-1][5] is None:
            rows[-1] = rows[-1][:5] + (lvg_ok,)

    # --- Centralized (top-level by-seed map; one pooled model, no per-client
    # SHAP/divergence, so those columns are n/a by design) ---
    cent = study.get("centralized_by_seed", {})
    for seed in SEEDS:
        entry = cent.get(str(seed))
        if entry is None:
            problems.append(f"centralized seed{seed}: MISSING (centralized_by_seed['{seed}'])")
            rows.append((f"centralized seed{seed}", False, False, False, False, False))
        else:
            rows.append((f"centralized seed{seed}", True, None, None, None, None))

    # --- Derived blocks ---
    nf_labels = {"fedavg"} | {f"fedprox_mu{m}" for m in MU_VALUES} | {"fedper", "local_only"}
    if any(r.get("aggregator") == "cluster_fl" for r in runs):
        nf_labels.add("cluster_fl")
    nf = study.get("preliminary_noise_floor", {})
    for label in sorted(nf_labels):
        if label not in nf:
            problems.append(f"preliminary_noise_floor missing {label}")
        else:
            rows.append((f"noise_floor:{label}", True, True, True, True, True))
    ds_labels = nf_labels | {"cluster_fl"}
    ds = study.get("divergence_summary", {})
    for label in sorted(ds_labels):
        if label not in ds:
            problems.append(f"divergence_summary missing {label}")
        else:
            rows.append((f"divergence_summary:{label}", True, True, True, True, True))

    # --- Cluster-FL stub documentation exists ---
    try:
        from src.config import CLUSTER_FL_RESULTS_PATH  # noqa: F401
    except ImportError:
        problems.append("config: CLUSTER_FL_RESULTS_PATH missing")
    if "clusters" not in CLUSTER_FL_RUN_SCHEMA:
        problems.append("CLUSTER_FL_RUN_SCHEMA missing 'clusters' field")

    # --- Matrix ---
    print("=" * 78)
    print(f"COMPLETENESS MATRIX - {STUDY_RESULTS_PATH}")
    print("=" * 78)
    print(f"{'Run':<34} | {'metrics':>7} | {'shap':>4} | {'diverg':>7} | {'scope':>5} | {'cmp':>4}")
    print("-" * 78)
    for label, m, s, d, sc, cp in rows:
        print(f"{label:<34} | {_cell(m):>7} | {_cell(s):>4} | {_cell(d):>7} | {_cell(sc):>5} | {_cell(cp):>4}")
    print("=" * 78)

    if problems:
        print(f"\nFAIL - {len(problems)} gap(s):")
        for p in problems:
            print(f"  - {p}")
        return 1
    print("\nPASS - all runs complete for both seeds; no gaps.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
