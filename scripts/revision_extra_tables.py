"""Generate tables and figures for the diagnostic experiments."""
from pathlib import Path
from statistics import mean

import matplotlib.pyplot as plt
import numpy as np

from scripts.make_all_tables_figures import fig_tex, fmt, read, save, summarize, table


def calibration(root):
    runs=[read(p) for p in sorted((root/"diagnostics").glob("seed*.json"))]
    if not runs:
        return [], None
    keys=("benign_recall","attack_recall","brier","ece","roc_auc","pr_auc")
    rows=[(m.replace("_"," "),*[summarize([r["variants"][m][k] for r in runs]) for k in keys])
          for m in runs[0]["variants"]]
    data=[table(f"Bluetooth diagnostics on {len(runs)} seeds, mean and sample standard deviation.",
                "newprobe",["Model","Specificity","Attack recall","Brier","ECE","ROC AUC","PR AUC"],rows)]
    run=runs[0]
    fig,axs=plt.subplots(2,2,figsize=(9,6))
    for m in ("fedavg","platt","temperature","frozen_trunk","fine_tune","fedper"):
        v=run["variants"][m]
        b=v["reliability"]
        axs[0,0].plot([x["score"] for x in b],[x["attack_fraction"] for x in b],"o-",markersize=3,label=m)
        axs[0,1].plot(v["roc"]["fpr"],v["roc"]["tpr"],label=m)
        axs[1,0].plot(v["pr"]["recall"],v["pr"]["precision"],label=m)
    axs[0,0].plot([0,1],[0,1],":",color="black")
    axs[0,0].set(xlabel="Mean predicted probability",ylabel="Observed attack fraction")
    axs[0,0].legend(fontsize=7)
    axs[0,1].set(xlabel="False-positive rate",ylabel="Attack recall")
    axs[1,0].set(xlabel="Attack recall",ylabel="Attack precision")
    with np.load(root/"diagnostics"/f"scores_{run['seed']}.npz") as p:
        for c,label in [(0,"Benign"),(1,"Attack")]:
            axs[1,1].hist(p["fedavg"][p["y_true"]==c],bins=30,density=True,alpha=.5,label=label)
    axs[1,1].set(xlabel="FedAvg logit",ylabel="Density")
    axs[1,1].legend()
    save(fig,"diagnostic")
    data.append(fig_tex("diagnostic",f"Bluetooth score diagnostics for seed {run['seed']}. Reliability, receiver operating characteristic, precision--recall, and the class-conditional FedAvg scores."))
    return data,runs


def shap_results(root):
    runs=[read(p) for p in sorted((root/"shap").glob("seed*.json"))]
    if not runs:
        return [], None
    names=("jaccard","spearman","cosine")
    rows=lambda d:[(k.replace("_"," "),*[summarize([x[k][m] for x in d if x[k][m] is not None])
                                        if any(x[k][m] is not None for x in d) else "--" for m in names]) for k in d[0]]
    within=table("FedPer versus FedAvg within client on matched backgrounds and records.", "newshapwithin",
                 ["Client and class","Top-five Jaccard","Spearman","Cosine"],rows([r["within_client"] for r in runs]))
    across=table("Within-method comparisons across clients using client-specific backgrounds.", "newshapacross",
                ["Pair and class","Top-five Jaccard","Spearman","Cosine"],rows([r["cross_client"]["fedavg"] for r in runs]))
    fig,ax=plt.subplots(figsize=(6,3.6))
    for method in runs[0]["cross_client"]:
        points=[]
        for r in runs:
            vals=[v["spearman"] for k,v in r["cross_client"][method].items()
                  if k.endswith("_attack") and v["spearman"] is not None]
            if not vals:
                continue
            confidence=read(root/"runs"/f"{method}_{r['seed']}.json")
            points.append([mean(vals),confidence["client_mean_macro_f1"]])
        if points:
            x,y=np.asarray(points).T
            ax.errorbar(x.mean(),y.mean(),xerr=x.std(ddof=1) if len(x)>1 else 0,
                        yerr=y.std(ddof=1) if len(y)>1 else 0,fmt="o",label=method.replace("_"," "))
    ax.set(xlabel="Mean attack attribution Spearman across client pairs",ylabel="Client-mean macro F1")
    ax.legend(fontsize=7,ncol=2)
    save(fig,"shap_tradeoff")
    figure=fig_tex("shap_tradeoff","Each point averages over the available matched seeds. The horizontal statistic first averages the three within-method protocol pairs. Error bars are sample standard deviations across seeds.")
    return [within,across,figure],runs


def supplementary(root):
    data=[]
    sensitivity=[read(p) for p in sorted((root/"sensitivity").glob("*.json"))
                 if not p.name.startswith("explanation_")]
    if sensitivity:
        keys=sorted({r["setting"] for r in sensitivity})
        rows=[]
        for k in keys:
            subset=[r for r in sensitivity if r["setting"]==k]
            rows.append((k.replace("_"," "),len(subset),summarize([r["mean_f1"] for r in subset]),
                         summarize([r["bluetooth_specificity"] for r in subset]),
                         summarize([r["bluetooth_shap_vs_full_fedavg"]["spearman"] for r in subset if r["bluetooth_shap_vs_full_fedavg"]["spearman"] is not None]) if any(r["bluetooth_shap_vs_full_fedavg"]["spearman"] is not None for r in subset) else "--"))
        data.append(table("One-factor settings and the round-by-epoch interaction on the reduced budget.",
                          "newsensitivity",["Setting","Seeds","Client mean macro F1","BT specificity","BT Spearman"],rows))
        fig,ax=plt.subplots(figsize=(8,5.5))
        for s in sorted({r["seed"] for r in sensitivity}):
            y={r["setting"]:r["bluetooth_specificity"] for r in sensitivity if r["seed"]==s}
            ax.scatter([y.get(k,np.nan) for k in keys],keys,label=str(s))
        ax.set(xlabel="Bluetooth benign recall")
        ax.legend(title="Seed")
        save(fig,"sensitivity")
        data.append(fig_tex("sensitivity","Bluetooth benign recall across tested sensitivity configurations."))
    scarcity=[read(p) for p in sorted((root/"scarcity").glob("*.json"))]
    if scarcity:
        fig,axes=plt.subplots(1,3,figsize=(10,3.2),sharey=True)
        rs=[r for runs in scarcity for r in runs["runs"]]
        for ax,k in zip(axes,("wifi","mqtt","bluetooth")):
            local=[r for r in rs if r["target"]==k]
            for m in sorted({r["method"] for r in local}):
                curves={r["budget"]:[] for r in local if r["method"]==m}
                for r in local:
                    if r["method"]==m: curves[r["budget"]].append(r["macro_f1"])
                x=sorted(curves);ax.plot(x,[mean(curves[i]) for i in x],"o-",markersize=3,label=m.replace("_"," "))
            ax.set(xscale="log",title=k,xlabel="Target training records")
        axes[0].set_ylabel("Macro F1")
        if rs:
            axes[0].legend(fontsize=6)
        save(fig,"scarcity")
        data.append(fig_tex("scarcity","Target-excluded adaptation by budget and protocol. Full budgets can differ by seed."))
    return data
