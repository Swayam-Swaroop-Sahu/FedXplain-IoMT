"""Print the manuscript evidence ledger. --check verifies the two detection tables and taxonomy."""
import argparse
import json
from pathlib import Path
from statistics import mean


def load(name):
    return json.loads((Path("results") / f"{name}.json").read_text())


def detection_rows(study, seed):
    rows = []
    candidates = [("Centralized", study["centralized_by_seed"][str(seed)]["client_metrics"])]
    for run in study["local_only"] + study["runs"]:
        if run["seed"] != seed:
            continue
        method = run.get("aggregator", "local_only")
        name = {"centralized":"Centralized","local_only":"Local-only","fedavg":"FedAvg","fedper":"FedPer",
                "cluster_fl":"Cluster-FL","fedprox":f"FedProx {run.get('mu')}"}[method]
        candidates.append((name,run["client_metrics"]))
    for name,metrics in candidates:
        clients = [metrics[k] for k in ("wifi","mqtt","bluetooth")]
        scores = [m["f1_macro"] for m in clients]
        values = scores + [mean(scores)] + [m["benign_recall"] for m in clients]
        rows.append(" & ".join([name]+[f"{x:.4f}" for x in values])+r"\\")
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check",action="store_true")
    args = parser.parse_args()
    study = load("study_results")
    tex = Path("access.tex").read_text(encoding="utf8")
    for seed in study["run_metadata"]["seeds"]:
        rows = detection_rows(study,seed)
        print("Detection, seed",seed,*rows,sep="\n")
        if args.check:
            part = tex.split(r"\label{tab:detect"+str(seed)+"}")[1].split(r"\end{table*}")[0]
            assert all(row in part for row in rows)
    for proto,seeds in load("attack_label_inventory").items():
        for name,counts in seeds["42"].items():
            n = sum(counts.values())
            print(proto,name,n)
            if args.check and name.lower() != "benign":
                assert name.replace("_",r"\_")+" & " in tex and f"{n:,}" in tex.split(name.replace("_",r"\_")+" & ")[1].split(r"\\")[0]
    for file in ("exact_match_analysis","calibration_probe","benign_shap_probe","sensitivity_analysis"):
        print(file,json.dumps(load(file),indent=2))
    for run in study["runs"]+study["local_only"]:
        print(run.get("aggregator","local_only"),run["seed"],run.get("mu"),json.dumps(run["divergence_full"]),json.dumps(run["jaccard_pairwise"]))
        if run.get("fedper_vs_fedavg"): print(json.dumps(run["fedper_vs_fedavg"]))
    gen = load("generalization_results")
    for group in ("zero_day","data_scarcity"):
        for target, result in gen[group]["targets"].items():
            print(group,target,json.dumps(result["summary"]))
    def record(section, cells, places=4):
        row = " & ".join(f"{v:.{places}f}" if isinstance(v, float) else str(v) for v in cells)
        print(section, row)
        if args.check:
            table = tex.split(r"\label{tab:" + section + "}")[1].split(r"\end{table")[0]
            assert row + r"\\" in table, (section, row)

    for r in load("calibration_probe")["runs"]:
        for model in ("baseline", "platt"):
            record("probe", [r["seed"], model, *[r[model][k] for k in
                   ("benign_recall", "attack_recall", "brier", "log_loss")]])
    for r in study["runs"]:
        if r["aggregator"] == "fedper":
            for client, v in r["fedper_vs_fedavg"].items():
                record("within", [client, r["seed"], v["top5_jaccard"], v["spearman"], v["cosine"]])
    for target, results in gen["zero_day"]["targets"].items():
        for method, metrics in results["summary"].items():
            record("transfer", [target, method.replace("_", "-"), metrics["mean_unseen_attack_recall"],
                                metrics["mean_detection_margin"]], places=3)
    for method in ("fedavg", "fedper", "cluster_fl", "local_only"):
        runs = [r for r in study["runs"] if r["aggregator"] == method] or study["local_only"]
        f1 = mean(v["f1_macro"] for r in runs for v in r["client_metrics"].values())
        recall = mean(r["client_metrics"]["bluetooth"]["benign_recall"] for r in runs)
        print(method, f"client mean F1 {f1:.4f}, Bluetooth specificity {recall:.4f}")
    for seed in (42, 7):
        def bt(method):
            runs = study["local_only"] if method == "local_only" else study["runs"]
            return next(r["client_metrics"]["bluetooth"]["benign_recall"] for r in runs
                        if r["seed"] == seed and r.get("aggregator","local_only") == method)
        avg, local, per = (bt(m) for m in ("fedavg","local_only","fedper"))
        print(seed, f"Bluetooth recovered gap {100*(per-avg)/(local-avg):.1f}%")
    print("Run configuration",study["run_metadata"])


if __name__ == "__main__":
    main()
