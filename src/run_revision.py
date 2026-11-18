"""Run the revision retraining suite. Existing reports are kept under results/revision."""
import argparse
from pathlib import Path
import json
from concurrent.futures import ProcessPoolExecutor
from itertools import product

import numpy as np
from scipy.special import expit
import torch

from src.revision_data import assign_split, prepare_clients, read_config, read_rows, split_report, write_manifest
from src.revision_metrics import metrics, label_recall
from src.revision_train import logits, train


MAIN = ["local_only","centralized","fedavg","fedprox","fedper","fixed_cluster","adaptive_cluster","centralized_matched"]
ABLATIONS = ["equal_weight","class_balanced"]


def baseline_grid(cfg):
    return [(m,mu) for m in MAIN+ABLATIONS for mu in (cfg["mu_values"] if m=="fedprox" else [0])]


def write_json(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value,indent=2,allow_nan=False)+"\n",encoding="utf8")
    temp.replace(path)


def get_data(cfg,seed,data="data/study",provenance=None,held_out=()):
    rows,features = read_rows(cfg["clients"],data,provenance)
    split = assign_split(rows,seed,cfg["validation_fraction"],cfg["test_fraction"])
    return split,features,prepare_clients(split,features,held_out)


def save_run(root,seed,tag,models,clients,cfg,metadata,snapshot=None):
    summary, all_y, all_scores = {}, [], []
    for client,model in models.items():
        part = clients[client]["test"]
        score = logits(model,part["x"])
        npz = root / "preds" / f"{tag}_{client}_{seed}.npz"
        npz.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(npz, sample_id=part["id"], y_true=part["y"], logit=score,
                            prob=expit(score.astype(float)), label=part["label"], group=part["group"])
        torch.save(model.state_dict(),root/"preds"/f"{tag}_{client}_{seed}.pth")
        if snapshot is not None:
            torch.save(snapshot[client],root/"preds"/f"{tag}_snapshot_{client}_{seed}.pth")
        summary[client] = metrics(part["y"],score,bins=cfg["reliability_bins"])
        summary[client]["per_source_label"] = label_recall(part["label"],score)
        all_y.append(part["y"])
        all_scores.append(score)
    metadata["client_metrics"] = summary
    metadata["client_mean_macro_f1"] = float(np.mean([v["macro_f1"] for v in summary.values()]))
    metadata["pooled_macro_f1"] = metrics(np.concatenate(all_y),np.concatenate(all_scores))["macro_f1"]
    write_json(root/"runs"/f"{tag}_{seed}.json",metadata)


def run_seed(job):
    seed,cfg,out,data,provenance,methods,held_out = job
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    root = Path(out)
    split,features,clients = get_data(cfg,seed,data,provenance,held_out)
    report = split_report(split,features)
    report["training_excluded_labels"] = list(held_out)
    report["assignment"] = write_manifest(split,Path("splits")/root.name,seed)
    write_json(root/"splits"/f"seed{seed}.json",report)
    for method,mu in baseline_grid(cfg):
        if method not in methods:
            continue
        tag = f"fedprox_{mu}" if method=="fedprox" else method
        if (root/"runs"/f"{tag}_{seed}.json").exists():
            continue
        models,logs,_ = train(clients,cfg,seed,method,mu)
        save_run(root,seed,tag,models,clients,cfg,dict(logs,seed=seed,method=method,mu=mu,config=cfg))
        print(seed,tag,round(logs["seconds"],2),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config",default="config/revision.yml")
    p.add_argument("--data",default="data/study")
    p.add_argument("--provenance")
    p.add_argument("--output",default="results/revision")
    p.add_argument("--seeds",nargs="+",type=int)
    p.add_argument("--methods",nargs="+",choices=MAIN+ABLATIONS,default=MAIN+ABLATIONS)
    p.add_argument("--exclude-label",action="append",default=[])
    p.add_argument("--workers",type=int,default=5)
    a=p.parse_args()
    cfg=read_config(a.config)
    jobs=[(seed,cfg,a.output,a.data,a.provenance,a.methods,a.exclude_label) for seed in (a.seeds or cfg["seeds"])]
    if a.workers==1:
        for job in jobs:
            run_seed(job)
    else:
        with ProcessPoolExecutor(max_workers=a.workers) as pool:
            list(pool.map(run_seed,jobs))


if __name__=="__main__":
    main()
