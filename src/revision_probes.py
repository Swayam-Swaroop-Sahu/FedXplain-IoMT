"""Validation-only diagnostics, class-conditional SHAP and offline attack transfer."""
import argparse
import copy
import hashlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from itertools import combinations

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import expit
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve, roc_curve
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
import torch

from src.run_revision import baseline_grid, get_data, write_json
from src.revision_metrics import metrics
from src.revision_data import read_config
from src.revision_train import as_tensor, fit, logits, make_model, optimizer


def load_model(root, cfg, name, client, seed):
    m = make_model(cfg)
    m.load_state_dict(torch.load(root/"preds"/f"{name}_{client}_{seed}.pth",weights_only=True))
    return m.eval()


def threshold_at_fpr(y,score,target):
    fpr,tpr,thresholds = roc_curve(y,score,drop_intermediate=False)
    ok = np.flatnonzero(fpr<=target)
    best = ok[np.argmax(tpr[ok])]
    return float(thresholds[best]) if np.isfinite(thresholds[best]) else float(np.nextafter(score.max(),np.inf))


def calibration(seed,cfg,bt,root):
    baseline = load_model(root,cfg,"fedavg","bluetooth",seed)
    fedper = load_model(root,cfg,"fedper","bluetooth",seed)
    local = load_model(root,cfg,"local_only","bluetooth",seed)
    v,t = bt["validation"],bt["test"]
    fit_ids,sel_ids = next(GroupShuffleSplit(n_splits=1,test_size=.5,random_state=seed).split(v["y"],groups=v["group"]))
    vscore,tscore = logits(baseline,v["x"]),logits(baseline,t["x"])
    y,score = v["y"][fit_ids],vscore[fit_ids]
    ysel = v["y"][sel_ids]
    logit_cut = threshold_at_fpr(ysel,vscore[sel_ids],cfg["operational_fpr"])
    lr = LogisticRegression().fit(score.reshape(-1,1),y)
    slope,intercept = lr.coef_.item(),lr.intercept_.item()
    temp = np.exp(minimize_scalar(lambda z: np.mean(np.logaddexp(0,score/np.exp(z))-y*score/np.exp(z)),
                                  bounds=(-4,4),method="bounded").x)
    assert np.array_equal(tscore>=0,tscore/temp>=0)
    variants = {"fedavg":(vscore,tscore,.5),"threshold":(vscore,tscore,float(expit(logit_cut))),
                "platt":(slope*vscore+intercept,slope*tscore+intercept,.5),
                "temperature":(vscore/temp,tscore/temp,.5),
                "fedper":(logits(fedper,v["x"]),logits(fedper,t["x"]),.5)}
    train_data = {k:v[k][fit_ids] for k in ("x","y")}
    for name in ("frozen_trunk","fine_tune"):
        model=copy.deepcopy(baseline)
        if name=="frozen_trunk":
            for p in list(model.parameters())[:4]:
                p.requires_grad_(False)
        opt=optimizer(model,cfg)
        best=None
        best_loss=float("inf")
        for r in range(10):
            fit(model,as_tensor(train_data),opt,cfg,1,seed+r)
            pred=logits(model,v["x"][sel_ids])
            loss=float(np.mean(np.logaddexp(0,pred)-ysel*pred))
            if loss<best_loss:
                best_loss=loss
                best=copy.deepcopy(model.state_dict())
        model.load_state_dict(best)
        variants[name]=(logits(model,v["x"]),logits(model,t["x"]),.5)
    result={}
    all_logits={}
    for name,(vlog,tlog,cut) in variants.items():
        out=metrics(t["y"],tlog,cut,cfg["reliability_bins"])
        limit=threshold_at_fpr(ysel,vlog[sel_ids],cfg["operational_fpr"])
        operational=metrics(t["y"],tlog,float(expit(limit)),cfg["reliability_bins"])
        out["validation_selected_operating_point"]={k:operational[k] for k in ("fpr","attack_recall","benign_recall")}
        out["threshold"]=cut
        fpr,tpr,_=roc_curve(t["y"],tlog)
        pp,rr,_=precision_recall_curve(t["y"],tlog)
        out["roc"]={"fpr":fpr.tolist(),"tpr":tpr.tolist()}
        out["pr"]={"precision":pp.tolist(),"recall":rr.tolist()}
        result[name]=out
        all_logits[name]=tlog
    out_dir=root/"diagnostics"
    out_dir.mkdir(exist_ok=True)
    np.savez_compressed(out_dir/f"scores_{seed}.npz",sample_id=t["id"],y_true=t["y"],**all_logits)
    def rep(m):
        with torch.no_grad():
            x=m.net[:5](torch.from_numpy(t["x"])).numpy().astype(float)
        return x-x.mean(axis=0)
    reps={k:rep(m) for k,m in {"fedavg":baseline,"fedper":fedper,"local_only":local}.items()}
    cka={a+"-"+b:float(np.linalg.norm(x.T@y)**2/(np.linalg.norm(x.T@x)*np.linalg.norm(y.T@y)))
         for a,b in combinations(reps,2) for x,y in [(reps[a],reps[b])]}
    write_json(out_dir/f"seed{seed}.json",dict(seed=seed,platt=[slope,intercept],temperature=float(temp),
        fit_ids=v["id"][fit_ids].tolist(),selection_ids=v["id"][sel_ids].tolist(),
        cka=cka,temperature_preserves_decisions=True,variants=result))


def similarity(a,b):
    top = lambda x: set(np.argsort(-x,kind="stable")[:5])
    aa,bb=top(a),top(b)
    denom=np.linalg.norm(a)*np.linalg.norm(b)
    rho=float(spearmanr(a,b).statistic) if np.ptp(a)>0 and np.ptp(b)>0 else None
    return {"jaccard":len(aa&bb)/len(aa|bb),"spearman":rho,
            "cosine":float(np.dot(a,b)/denom) if denom else None}


def explanation(seed,cfg,clients,root):
    import shap
    rng=np.random.default_rng(seed)
    draws={}
    for k,parts in clients.items():
        train,test=parts["train"],parts["test"]
        benign=np.flatnonzero(train["y"]==0)
        bg=rng.choice(benign,min(len(benign),cfg["shap_background"]),replace=False)
        sets={}
        boot={}
        for label,class_name in [(0,"benign"),(1,"attack")]:
            allowed=np.flatnonzero(test["y"]==label)
            sets[class_name]=rng.choice(allowed,min(len(allowed),cfg["shap_samples"]),replace=False)
            n=len(sets[class_name])
            boot[class_name]=rng.integers(0,n,(cfg["shap_bootstrap"],n))
        draws[k]=(bg,sets,boot)
    vectors,values={},{}
    for method in ("fedavg","fedper","fixed_cluster","adaptive_cluster","local_only","centralized"):
        vectors[method]={}
        for k,parts in clients.items():
            bg,sets,boot=draws[k]
            model=load_model(root,cfg,method,k,seed)
            explainer=shap.DeepExplainer(model,torch.from_numpy(parts["train"]["x"][bg]))
            vectors[method][k]={}
            for name,ix in sets.items():
                phi=np.asarray(explainer.shap_values(torch.from_numpy(parts["test"]["x"][ix]))).reshape(len(ix),-1)
                values[f"{method}_{k}_{name}"]=phi
                vectors[method][k][name]={"importance":np.abs(phi).mean(axis=0).tolist(),
                    "background_ids":parts["train"]["id"][bg].tolist(),"sample_ids":parts["test"]["id"][ix].tolist()}
    def compare(a,b,k,j,name):
        x,y=(np.abs(values[f"{method}_{client}_{name}"]) for method,client in [(a,k),(b,j)])
        result=similarity(x.mean(axis=0),y.mean(axis=0))
        paired=zip(draws[k][2][name],draws[j][2][name])
        boot=[similarity(x[i].mean(axis=0),y[h].mean(axis=0)) for i,h in paired]
        result["sample_bootstrap_95"]={m:np.quantile([b[m] for b in boot if b[m] is not None],[.025,.975]).tolist()
                                       if any(b[m] is not None for b in boot) else None for m in result.copy()}
        return result
    within={f"{k}_{name}":compare("fedavg","fedper",k,k,name)
            for k in clients for name in ("benign","attack")}
    across={m:{f"{a}-{b}_{name}":compare(m,m,a,b,name) for a,b in combinations(clients,2)
               for name in ("benign","attack")} for m in vectors}
    path=root/"shap"
    path.mkdir(exist_ok=True)
    np.savez_compressed(path/f"samples_{seed}.npz",**values)
    write_json(path/f"seed{seed}.json",dict(seed=seed,shap_version=shap.__version__,vectors=vectors,
                                         within_client=within,cross_client=across))


def transfer(seed,cfg,rows,features,root):
    outcome={}
    methods=[f"fedprox_{mu}" if m=="fedprox" else m for m,mu in baseline_grid(cfg)]
    for client in cfg["clients"]:
        blocks=rows[rows["client"].eq(client)]
        local_train=blocks[blocks["split"].eq("train")]
        scaler=StandardScaler().fit(local_train[features])
        seen=set(local_train["label"])
        result={}
        for source in cfg["clients"]:
            if source==client:
                continue
            foreign=rows[rows["client"].eq(source)&rows["split"].eq("test")]
            benign=foreign["y"].eq(0).to_numpy()
            x=scaler.transform(foreign[features]).astype(np.float32)
            labels=foreign["label"].to_numpy()
            result[source]={}
            for m in methods:
                model=load_model(root,cfg,m,client,seed)
                decisions=logits(model,x)>=0
                fpr=float(decisions[benign].mean())
                result[source][m]={"benign_fpr":fpr,"benign_support":int(benign.sum()),"attack":{
                    k:{"recall":float(decisions[mask].mean()),"support":int(mask.sum()),
                       "margin":float(decisions[mask].mean()-fpr)}
                    for k in sorted(set(labels)-seen-{"Benign"}) if (mask:=labels==k).any()}}
        outcome[client]=result
    write_json(root/"transfer"/f"seed{seed}.json",outcome)


def job(args):
    seed,cfg,root,tasks=args
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    rows,features,clients=get_data(cfg,seed)
    for task in tasks:
        if task=="calibration": calibration(seed,cfg,clients["bluetooth"],root)
        if task=="shap": explanation(seed,cfg,clients,root)
        if task=="transfer": transfer(seed,cfg,rows,features,root)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",default="results/revision")
    p.add_argument("--seeds",nargs="+",type=int)
    p.add_argument("--tasks",nargs="+",choices=["calibration","shap","transfer"],
                   default=["calibration","shap","transfer"])
    a=p.parse_args()
    cfg=read_config()
    seeds=a.seeds or cfg["seeds"]
    with ProcessPoolExecutor(max_workers=min(3,len(seeds))) as pool:
        list(pool.map(job,[(s,cfg,Path(a.output),a.tasks) for s in seeds]))


if __name__=="__main__":
    main()
