"""Target-excluded pretraining and matched-step adaptation at seven budgets."""
import argparse
import copy
from concurrent.futures import ProcessPoolExecutor
import math
from pathlib import Path

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit
import torch

from src.revision_data import read_config
from src.revision_train import as_tensor, fit, logits, make_model, optimizer, train
from src.revision_metrics import metrics
from src.run_revision import get_data, write_json


def job(args):
    seed,cfg,target,root=args
    path=Path(root)/"scarcity"/f"{target}_{seed}.json"
    if path.exists():
        return
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    rows,features,clients=get_data(cfg,seed)
    original=clients[target]
    source={k:v for k,v in clients.items() if k!=target}
    models,log,_=train(source,cfg,seed,"fedavg")
    pretrain=next(iter(models.values()))
    data,test=original["train"],original["test"]
    train_raw=rows.loc[rows["client"].eq(target) & rows["split"].eq("train"),features].to_numpy()
    test_raw=rows.loc[rows["client"].eq(target) & rows["split"].eq("test"),features].to_numpy()
    out=[]
    steps=200
    for n in [50,100,250,500,1000,5000,len(data["y"])]:
        for rep in range(3):
            sample_seed=seed*1000+rep
            idx=slice(None) if n==len(data["y"]) else next(StratifiedShuffleSplit(
                    n_splits=1,train_size=n,random_state=sample_seed).split(data["x"],data["y"]))[0]
            part={k:data[k][idx] for k in ("x","y","id")}
            scaler=StandardScaler().fit(train_raw[idx])
            part["x"]=scaler.transform(train_raw[idx]).astype(np.float32)
            x_test=scaler.transform(test_raw).astype(np.float32)
            # Fixed compute per arm under each budget. All arms use the same sampled IDs.
            epochs=math.ceil(steps/math.ceil(n/cfg["batch_size"]))
            torch.manual_seed(sample_seed)
            initial=make_model(cfg)
            for method in ("scratch","frozen_new_head","fedper_style","full_fine_tune"):
                model=copy.deepcopy(initial if method=="scratch" else pretrain)
                if method in ("frozen_new_head","fedper_style"):
                    model.net[5]=copy.deepcopy(initial.net[5])
                if method=="frozen_new_head":
                    for p in list(model.parameters())[:4]:
                        p.requires_grad_(False)
                if method=="fedper_style":
                    # Personalize both the last hidden layer and the head.
                    for p in list(model.parameters())[:2]:
                        p.requires_grad_(False)
                stat=fit(model,as_tensor(part),optimizer(model,cfg),cfg,epochs,sample_seed,steps=steps)
                result=metrics(test["y"],logits(model,x_test))
                out.append(dict(method=method,seed=seed,target=target,budget=n,repeat=rep,**stat,
                    **{k:result[k] for k in ("macro_f1","benign_recall","attack_recall")}))
    write_json(path,dict(pretraining_clients=list(source),pretraining_selected_round=log["selected_round"],
        adaptation_scaler="Fit on the sampled target rows only",runs=out))
    print("scarcity",target,seed,flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root",default="results/revision")
    a=p.parse_args()
    cfg=read_config()
    jobs=[(s,cfg,t,a.root) for s in cfg["seeds"][:2] for t in cfg["clients"]]
    with ProcessPoolExecutor(max_workers=3) as pool:
        list(pool.map(job,jobs))


if __name__=="__main__":
    main()
