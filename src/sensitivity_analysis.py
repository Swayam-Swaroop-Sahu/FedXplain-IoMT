"""Reorganize recorded parameter contrasts. Run python -m src.sensitivity_analysis."""
import json
from pathlib import Path
from statistics import mean

from src import config


def main():
    study = json.loads(Path(config.STUDY_RESULTS_PATH).read_text())
    groups = {}
    for run in study["runs"] + study["local_only"]:
        name = run.get("aggregator", "local_only")
        if name == "fedprox":
            name += f"_{run['mu']}"
        groups.setdefault(name, {})[str(run["seed"])] = run["client_metrics"]
    groups["centralized"] = {s: r["client_metrics"] for s, r in study["centralized_by_seed"].items()}
    paired = {}
    for method, runs in groups.items():
        paired[method] = {client: {key: {"42": a, "7": runs["7"][client][key],
                                          "absolute_difference": abs(a - runs["7"][client][key])}
                                  for key, a in metrics.items()}
                          for client, metrics in runs["42"].items()}
    sweep = []
    for mu in study["run_metadata"]["mu_values"]:
        pair = paired[f"fedprox_{mu}"]["bluetooth"]["benign_recall"]
        sweep.append({"mu": mu, "42": pair["42"], "7": pair["7"], "mean": mean([pair["42"], pair["7"]])})
    meta = study["run_metadata"]
    fixed = {"rounds": [meta["n_rounds"], "Common communication budget"], "local_epochs": [meta["n_local_epochs"], "Common local workload per round"],
             "batch_size": [config.BATCH_SIZE, "Shared minibatch size"], "learning_rate": [config.LEARNING_RATE, "Common Adam step size"],
             "hidden_widths": [[64, 32], "Common feature extractor across methods"],
             "dropout": [0.2, "Regularization during training"], "shap_background": [meta["shap_bg_size"], "Bound explanation cost"],
             "shap_examples": [meta["shap_attack_size"], "Common client explanation budget"]}
    result = {"seeds": [42, 7], "seed_design": "Matched split and training seeds; both factors change together.",
              "penalty_design": "Four positive strengths test increasing anchoring under the same budget.",
              "fixed_settings": fixed, "fixed_setting_effects": "Unidentified from the saved runs.",
              "client_metric_pairs": paired, "bluetooth_penalty_sweep": sweep,
              "explanation_metric_pairs": study["divergence_summary"]}
    Path("results/sensitivity_analysis.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
