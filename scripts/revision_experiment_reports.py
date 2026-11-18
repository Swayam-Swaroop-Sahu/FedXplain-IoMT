"""Summarize completed E1–E10 artifacts and compute budget for the revision."""
from collections import defaultdict
import csv
import json
from pathlib import Path
from statistics import mean


def read(path):
    return json.loads(path.read_text())


def main():
    root=Path("results/revision")
    grouped={}
    for name,path in [("primary",root),("withheld",Path("results/revision_holdout"))]:
        grouped[name]=[read(p) for p in sorted((path/"runs").glob("*.json"))]
    data=grouped["primary"]
    assert len(data)==70 and len(grouped["withheld"])==28
    runtime={k:{"training_seconds_sum":sum(x["seconds"] for x in v),"runs":len(v),
                "steps":sum(x["optimizer_steps"] for x in v)} for k,v in grouped.items()}
    plateau=[]
    for run in data:
        values=[mean(h["validation_bce"].values()) for h in run["history"]]
        # Require at least three evaluations, with the entire tail within tolerance.
        first=next((i+1 for i in range(len(values)-2) if max(values[i:])-min(values[i:])<=0.001),None)
        plateau.append({k:run[k] for k in ("seed","method","mu","selected_round","optimizer_steps")}|
                       {"first_round_within_0.001_for_remainder":first})
    (root/"training_summary.json").write_text(json.dumps({"compute":runtime,"plateau":plateau},indent=2)+"\n")
    visibility=[]
    for protocol,path in [("standard",root),("withheld",Path("results/revision_holdout"))]:
        for file in sorted((path/"splits").glob("seed*.json")):
            r=read(file)
            seen=defaultdict(dict)
            for x in r["counts"]:
                seen[(x["client"],x["label"])][x["split"]]=x["rows"]
            for (k,l),splits in seen.items():
                visibility.append(dict(protocol=protocol,seed=file.stem[4:],client=k,label=l,
                    train=splits.get("train",0) if l not in r["training_excluded_labels"] else 0,
                    validation=splits.get("validation",0) if l not in r["training_excluded_labels"] else 0,
                    test=splits.get("test",0)))
    with (root/"training_visibility.csv").open("w",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(visibility[0]))
        writer.writeheader()
        writer.writerows(visibility)
    fedavg=[r for r in data if r["method"]=="fedavg"]
    methods=lambda name:[r for r in data if r["method"]==name]
    bt=lambda runs:mean(r["client_metrics"]["bluetooth"]["benign_recall"] for r in runs)
    diagnostics=[read(p) for p in (root/"diagnostics").glob("seed*.json")]
    cka=mean(r["cka"]["fedavg-fedper"] for r in diagnostics) if diagnostics else None
    e2=f"FedAvg/FedPer Bluetooth trunk mean linear CKA was {cka:.4f}. Validation-only output and parameter variants are in diagnostics/." if cka is not None else "No completed diagnostics."
    hidden=[r for r in grouped["withheld"] if r["method"]=="fedavg"]
    withheld_recall=mean(v["recall"] for r in hidden for k,v in r["client_metrics"]["wifi"]["per_source_label"].items() if k.startswith("Recon"))
    results={
        "E01":f"Five seeds and 14 configurations. FedAvg Bluetooth benign recall averaged {bt(fedavg):.4f}, FedPer {bt(methods('fedper')):.4f}, fixed groups {bt(methods('fixed_cluster')):.4f}, adaptive groups {bt(methods('adaptive_cluster')):.4f}, and local-only {bt(methods('local_only')):.4f}.",
        "E02":e2+f" Equal weighting obtained {bt(methods('equal_weight')):.4f} specificity and class balancing {bt(methods('class_balanced')):.4f}. Client update cosines are archived in runs/.",
        "E03":"See sensitivity/ for training and SHAP perturbations on seeds 42 and 7. The five-seed main series varies the proximal coefficient.",
        "E04":f"Training used {runtime['primary']['steps']:,} optimizer steps across the main suite. See training_summary.json for the explicit descriptive plateau rule. Pooled matched models used the same number of steps as FedAvg.",
        "E05":f"Source-matched margins are in transfer/. Omitted reconnaissance labels were excluded from training and validation for all 28 holdout models. FedAvg reached a mean label recall of {withheld_recall:.4f} on these labels. This is one condition on a derived dataset.",
        "E06":"Scarcity reports use a source-only FedAvg trunk, seven budgets and three repetitions on each of seeds 42 and 7. The same 100 optimizer steps and target subsets are used by all arms.",
        "E07":"Per-label support, attack-as-positive recall and precision against benign-only controls are in per_label_metrics.csv. The detector assigns one binary class and does not predict an attack family.",
        "E08":"The shap/ directory contains paired background/sample IDs, class-conditional local attributions, within-client and cross-client comparisons and sample-bootstrap intervals.",
        "E09":"Optional drift-correction baselines were outside the completed compute budget.",
        "E10":"The paper makes no inference latency or deployment-efficiency claim."
    }
    for key,value in results.items():
        (root/f"{key}_summary.md").write_text(f"# {key}\n\n{value}\n",encoding="utf8")


if __name__=="__main__":
    main()
