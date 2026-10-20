"""JSON schema and integrity validator for study_results.json.

Validates:
- No NaN, Infinity, or -Infinity values in the JSON.
- All top-level keys exist and have valid structure.
- Every run (centralized, FedAvg, FedProx, FedPer, local-only) has all required
  keys and correct types.
- Plausible value ranges:
  - Metrics (precision, recall, f1, accuracy) in [0.0, 1.0].
  - Confusion matrix counts (tn, fp, fn, tp) are non-negative ints.
  - Jaccard indices in [0.0, 1.0].
  - Spearman correlation and cosine similarity in [-1.0, 1.0] (or None if uncomputable).
  - Feature vectors have exactly 45 features with consistent names and order.
  - Mean |SHAP| values are non-negative.

Exit code 0 on success, 1 on any schema or integrity violation.
"""

import json
import math
import os
import sys

# Ensure repository root is in sys.path
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.config import EXPECTED_FEATURE_COUNT, PROTOCOLS, SEEDS, STUDY_RESULTS_PATH
from src.run_study_experiments import PAIR_LABELS

METRIC_BOUNDS = {
    "f1": (0.0, 1.0),
    "f1_binary": (0.0, 1.0),
    "f1_macro": (0.0, 1.0),
    "f1_weighted": (0.0, 1.0),
    "accuracy": (0.0, 1.0),
    "benign_precision": (0.0, 1.0),
    "benign_recall": (0.0, 1.0),
    "attack_precision": (0.0, 1.0),
    "attack_recall": (0.0, 1.0),
}


def _check_no_nan_inf(obj, path="root", errors=None):
    if errors is None:
        errors = []
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            errors.append(f"Non-finite float at {path}: {obj}")
    elif isinstance(obj, dict):
        for k, v in obj.items():
            _check_no_nan_inf(v, f"{path}.{k}", errors)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            _check_no_nan_inf(v, f"{path}[{i}]", errors)
    return errors


def _validate_client_metrics(client_metrics, label, errors):
    if not isinstance(client_metrics, dict):
        errors.append(f"{label}: client_metrics must be a dict")
        return
    for proto in PROTOCOLS:
        if proto not in client_metrics:
            errors.append(f"{label}: missing client_metrics for {proto}")
            continue
        m = client_metrics[proto]
        if not isinstance(m, dict):
            errors.append(f"{label}.{proto}: metrics must be a dict")
            continue
        for key, (low, high) in METRIC_BOUNDS.items():
            if key not in m:
                errors.append(f"{label}.{proto}: missing metric key '{key}'")
            else:
                val = m[key]
                if not isinstance(val, (int, float)):
                    errors.append(f"{label}.{proto}.{key}: expected number, got {type(val).__name__}")
                elif not (low - 1e-6 <= val <= high + 1e-6):
                    errors.append(f"{label}.{proto}.{key}: {val} out of bounds [{low}, {high}]")
        for count_key in ("tn", "fp", "fn", "tp", "n_benign", "n_attack"):
            if count_key not in m:
                errors.append(f"{label}.{proto}: missing count key '{count_key}'")
            else:
                cval = m[count_key]
                if not isinstance(cval, int) or cval < 0:
                    errors.append(f"{label}.{proto}.{count_key}: expected non-negative int, got {cval}")


CENTRALIZED_METRIC_BOUNDS = {
    "f1_binary": (0.0, 1.0),
    "f1_macro": (0.0, 1.0),
    "f1_weighted": (0.0, 1.0),
    "accuracy": (0.0, 1.0),
    "benign_precision": (0.0, 1.0),
    "benign_recall": (0.0, 1.0),
    "attack_precision": (0.0, 1.0),
    "attack_recall": (0.0, 1.0),
}


def _validate_centralized_metrics(cent_entry, label, errors):
    if not isinstance(cent_entry, dict):
        errors.append(f"{label}: entry must be a dict")
        return
    for key, (low, high) in CENTRALIZED_METRIC_BOUNDS.items():
        if key not in cent_entry:
            errors.append(f"{label}: missing metric key '{key}'")
        else:
            val = cent_entry[key]
            if not isinstance(val, (int, float)):
                errors.append(f"{label}.{key}: expected number, got {type(val).__name__}")
            elif not (low - 1e-6 <= val <= high + 1e-6):
                errors.append(f"{label}.{key}: {val} out of bounds [{low}, {high}]")
    cm = cent_entry.get("confusion_matrix")
    if not isinstance(cm, dict):
        errors.append(f"{label}: confusion_matrix must be a dict")
    else:
        for count_key in ("tn", "fp", "fn", "tp"):
            if count_key not in cm:
                errors.append(f"{label}.confusion_matrix: missing '{count_key}'")
            else:
                cval = cm[count_key]
                if not isinstance(cval, int) or cval < 0:
                    errors.append(f"{label}.confusion_matrix.{count_key}: expected non-negative int, got {cval}")


def _validate_shap_data(run, label, errors, expected_feature_names=None):
    top10 = run.get("shap_top10_per_client")
    top5 = run.get("shap_top5_per_client")
    full_vecs = run.get("shap_full_vectors_per_client")

    if not isinstance(top10, dict):
        errors.append(f"{label}: shap_top10_per_client must be a dict")
    if not isinstance(top5, dict):
        errors.append(f"{label}: shap_top5_per_client must be a dict")
    if not isinstance(full_vecs, dict):
        errors.append(f"{label}: shap_full_vectors_per_client must be a dict")
        return

    for proto in PROTOCOLS:
        t10 = top10.get(proto, [])
        if len(t10) != 10:
            errors.append(f"{label}.{proto}: shap_top10 has {len(t10)} items, expected 10")
        for item in t10:
            if not isinstance(item, list) or len(item) != 2:
                errors.append(f"{label}.{proto}: invalid top10 item {item}")
            else:
                feat, val = item
                if not isinstance(feat, str) or not isinstance(val, (int, float)) or val < 0:
                    errors.append(f"{label}.{proto}: invalid top10 entry {item}")

        t5 = top5.get(proto, [])
        if len(t5) != 5:
            errors.append(f"{label}.{proto}: shap_top5 has {len(t5)} items, expected 5")
        if not all(isinstance(f, str) for f in t5):
            errors.append(f"{label}.{proto}: top5 items must be strings")

        vec = full_vecs.get(proto, {})
        names = vec.get("feature_names", [])
        values = vec.get("values", [])
        if len(names) != EXPECTED_FEATURE_COUNT:
            errors.append(f"{label}.{proto}: feature_names length {len(names)} != {EXPECTED_FEATURE_COUNT}")
        if len(values) != EXPECTED_FEATURE_COUNT:
            errors.append(f"{label}.{proto}: values length {len(values)} != {EXPECTED_FEATURE_COUNT}")
        if not all(isinstance(v, (int, float)) and v >= 0 for v in values):
            errors.append(f"{label}.{proto}: all full vector values must be non-negative floats")
        if expected_feature_names and names != expected_feature_names:
            errors.append(f"{label}.{proto}: feature_names order does not match standard order")


def _validate_divergence(run, label, errors):
    jaccard = run.get("jaccard_pairwise", {})
    div_full = run.get("divergence_full", {})
    shared_feats = run.get("shared_features", {})

    for pair in PAIR_LABELS:
        if pair not in jaccard:
            errors.append(f"{label}: jaccard_pairwise missing {pair}")
        else:
            j = jaccard[pair]
            if not isinstance(j, (int, float)) or not (0.0 <= j <= 1.0):
                errors.append(f"{label}: jaccard[{pair}] {j} out of [0, 1]")

        if pair not in div_full:
            errors.append(f"{label}: divergence_full missing {pair}")
        else:
            d = div_full[pair]
            sp = d.get("spearman")
            cs = d.get("cosine")
            if sp is not None and (not isinstance(sp, (int, float)) or not (-1.0 <= sp <= 1.0)):
                errors.append(f"{label}: divergence_full[{pair}].spearman {sp} out of [-1, 1]")
            if cs is not None and (not isinstance(cs, (int, float)) or not (-1.0 <= cs <= 1.0)):
                errors.append(f"{label}: divergence_full[{pair}].cosine {cs} out of [-1, 1]")

        if pair not in shared_feats:
            errors.append(f"{label}: shared_features missing {pair}")
        elif not isinstance(shared_feats[pair], list):
            errors.append(f"{label}: shared_features[{pair}] must be a list")


def validate_results_file(results_path=STUDY_RESULTS_PATH) -> tuple[bool, list[str]]:
    errors = []
    if not os.path.exists(results_path):
        return False, [f"File not found: {results_path}"]

    with open(results_path, "r", encoding="utf-8") as f:
        # Check raw JSON parse first
        try:
            raw_text = f.read()
            study = json.loads(raw_text)
        except Exception as e:
            return False, [f"JSON parse error: {e}"]

    # 1. Check for NaN / Infinity
    _check_no_nan_inf(study, errors=errors)

    # 2. Check top-level keys
    required_top_keys = [
        "run_metadata", "centralized", "centralized_by_seed", "poc_baseline_reference",
        "runs", "local_only", "local_vs_global", "preliminary_noise_floor", "divergence_summary"
    ]
    for k in required_top_keys:
        if k not in study:
            errors.append(f"Missing top-level key: {k}")

    # Standard feature names reference
    standard_features = None
    if "runs" in study and len(study["runs"]) > 0:
        first_run = study["runs"][0]
        standard_features = first_run.get("shap_full_vectors_per_client", {}).get("wifi", {}).get("feature_names")
        if standard_features and len(standard_features) != EXPECTED_FEATURE_COUNT:
            errors.append(f"Standard features count {len(standard_features)} != {EXPECTED_FEATURE_COUNT}")

    # 3. Check centralized runs
    if "centralized" in study:
        _validate_centralized_metrics(study["centralized"], "centralized", errors)
    cent_by_seed = study.get("centralized_by_seed", {})
    for seed in SEEDS:
        s_key = str(seed)
        if s_key not in cent_by_seed:
            errors.append(f"centralized_by_seed missing seed '{s_key}'")
        else:
            _validate_centralized_metrics(cent_by_seed[s_key], f"centralized_seed{seed}", errors)

    # 4. Check runs array (federated & personalized)
    runs = study.get("runs", [])
    seen_runs = set()
    for idx, r in enumerate(runs):
        agg = r.get("aggregator")
        seed = r.get("seed")
        mu = r.get("mu")
        run_id = (agg, seed, mu)
        label = f"run[{idx}]: {agg}_seed{seed}_mu{mu}"
        seen_runs.add(run_id)

        if seed not in SEEDS:
            errors.append(f"{label}: unexpected seed {seed}")
        if agg not in ("fedavg", "fedprox", "fedper", "cluster_fl"):
            errors.append(f"{label}: unknown aggregator {agg}")

        _validate_client_metrics(r.get("client_metrics", {}), label, errors)
        _validate_shap_data(r, label, errors, standard_features)
        _validate_divergence(r, label, errors)

        if agg == "fedper":
            if "avg_macro_f1" not in r or not (0.0 <= r["avg_macro_f1"] <= 1.0):
                errors.append(f"{label}: invalid avg_macro_f1")
            if "fedper_vs_fedavg" not in r:
                errors.append(f"{label}: missing fedper_vs_fedavg")
            if "fedper_vs_local_only" not in r:
                errors.append(f"{label}: missing fedper_vs_local_only")

    # 5. Check local-only runs
    local_runs = study.get("local_only", [])
    for idx, r in enumerate(local_runs):
        seed = r.get("seed")
        label = f"local_only[{idx}]: seed{seed}"
        if seed not in SEEDS:
            errors.append(f"{label}: unexpected seed {seed}")
        _validate_client_metrics(r.get("client_metrics", {}), label, errors)
        _validate_shap_data(r, label, errors, standard_features)
        _validate_divergence(r, label, errors)

    # 6. Check local_vs_global
    lvg = study.get("local_vs_global", {})
    for seed in SEEDS:
        key = f"seed{seed}_local_vs_fedavg_seed{seed}"
        if key not in lvg:
            errors.append(f"local_vs_global missing '{key}'")
        else:
            entry = lvg[key]
            for proto in PROTOCOLS:
                if proto not in entry:
                    errors.append(f"local_vs_global.{key} missing proto {proto}")
                else:
                    pd_data = entry[proto]
                    tj = pd_data.get("top5_jaccard")
                    if not isinstance(tj, (int, float)) or not (0.0 <= tj <= 1.0):
                        errors.append(f"local_vs_global.{key}.{proto}.top5_jaccard {tj} out of [0, 1]")

    # 7. Check preliminary_noise_floor and divergence_summary
    nf = study.get("preliminary_noise_floor", {})
    for agg_key in ("fedavg", "fedper", "local_only"):
        if agg_key not in nf:
            errors.append(f"preliminary_noise_floor missing {agg_key}")

    ds = study.get("divergence_summary", {})
    if "cluster_fl" not in ds:
        errors.append("divergence_summary missing cluster_fl")
    else:
        cfl = ds["cluster_fl"]
        if "status" in cfl:
            if cfl["status"] != "slot empty - cluster_fl results pending":
                errors.append(f"divergence_summary cluster_fl unexpected status: {cfl}")
        else:
            for seed in SEEDS:
                skey = f"seed{seed}"
                if skey not in cfl:
                    errors.append(f"divergence_summary.cluster_fl missing {skey}")

    return (len(errors) == 0, errors)


def main():
    print(f"Validating {STUDY_RESULTS_PATH}...")
    success, errors = validate_results_file()
    if not success:
        print(f"\n[FAIL] Found {len(errors)} validation errors:")
        for e in errors[:50]:
            print(f"  - {e}")
        if len(errors) > 50:
            print(f"  ... and {len(errors) - 50} more errors.")
        sys.exit(1)
    else:
        print("\n[PASS] All schema and integrity checks passed successfully (0 errors).")
        sys.exit(0)


if __name__ == "__main__":
    main()
