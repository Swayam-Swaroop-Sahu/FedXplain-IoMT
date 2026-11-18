"""One-factor and round/epoch interaction sweeps on a two-seed compute budget."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch

from src.revision_data import read_config
from src.revision_metrics import metrics
from src.revision_probes import explanation, load_model, similarity
from src.revision_train import logits, train
from src.run_revision import get_data, write_json


def grid():
    return {"anchor":{}, "rounds_2":{"rounds":2},"rounds_6":{"rounds":6},"epochs_1":{"local_epochs":1},
        "epochs_3":{"local_epochs":3},"lr_low":{"lr":.0005},"lr_high":{"lr":.002},
        "batch_32":{"batch_size":32},"batch_128":{"batch_size":128},"dropout_0":{"dropout":0},
        "dropout_04":{"dropout":.4},"width_half":{"widths":[32,16]},"width_double":{"widths":[128,64]},
        "interaction_low":{"rounds":2,"local_epochs":1},"interaction_high":{"rounds":6,"local_epochs":3},
        "head_1":{},"head_2":{}}


def job(job_args):
    import shap
    seed,cfg,root=job_args
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    _,_,clients=get_data(cfg,seed)
    rng=np.random.default_rng(seed)
    parts=clients["bluetooth"]
    b=rng.permutation(np.flatnonzero(parts["train"]["y"]==0))
    t=rng.permutation(np.flatnonzero(parts["test"]["y"]==1))
    def attributed(model,n_background=40,n_explained=100):
        exp=shap.DeepExplainer(model.eval(),torch.from_numpy(parts["train"]["x"][b[:n_background]]))
        phi=np.asarray(exp.shap_values(torch.from_numpy(parts["test"]["x"][t[:n_explained]])))
        return np.abs(phi).reshape(len(t[:n_explained]),-1).mean(axis=0)
    out=Path(root)/"sensitivity"
    anchor_cfg=dict(cfg,rounds=4,local_epochs=2)
    ref_model=load_model(Path(root),cfg,"fedavg","bluetooth",seed)
    ref=attributed(ref_model)
    for key,overrides in grid().items():
        path=out/f"{key}_{seed}.json"
        if path.exists():
            continue
        settings=dict(anchor_cfg,**overrides)
        method="fedper" if key.startswith("head") else "fedavg"
        models,log,_=train(clients,settings,seed,method,head_depth=2 if key=="head_2" else 1)
        values={k:metrics(v["test"]["y"],logits(models[k],v["test"]["x"])) for k,v in clients.items()}
        write_json(path,dict(seed=seed,setting=key,method=method,config=settings,
            mean_f1=float(np.mean([v["macro_f1"] for v in values.values()])),
            bluetooth_specificity=values["bluetooth"]["benign_recall"],
            bluetooth_shap_vs_full_fedavg=similarity(attributed(models["bluetooth"]),ref),**log))
        print("sensitivity",seed,key,flush=True)
    per=load_model(Path(root),cfg,"fedper","bluetooth",seed)
    attr={}
    for bg,n in [(20,100),(40,100),(100,100),(40,50),(40,200)]:
        attr[f"{bg}_{n}"]=similarity(attributed(ref_model,bg,n),attributed(per,bg,n))
    write_json(out/f"explanation_budget_{seed}.json",attr)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root",default="results/revision")
    a=p.parse_args()
    cfg=read_config()
    with ProcessPoolExecutor(max_workers=2) as pool:
        list(pool.map(job,[(s,cfg,a.root) for s in cfg["seeds"][:2]]))


if __name__=="__main__":
    main()
