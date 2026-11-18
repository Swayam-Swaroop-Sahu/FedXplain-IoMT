"""Draw study-specific diagrams and observed class prevalence from the label inventory."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def box(ax, x, y, label, color="#e2f1ef"):
    ax.text(x,y,label,ha="center",va="center",fontsize=11,
            bbox=dict(boxstyle="round,pad=0.65", fc=color, ec="#315069"))


def arrow(ax, start, end):
    ax.annotate("",xy=end,xytext=start,arrowprops=dict(arrowstyle="->",color="#345c77",lw=1.7))


def save(fig, name):
    fig.savefig(Path("figures") / f"fig_{name}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    counts=json.loads(Path("results/attack_label_inventory.json").read_text())
    names=["Wi-Fi","MQTT","Bluetooth"]
    partitions=[c["42"] for c in counts.values()]
    sizes=[sum(v["train"] for v in labels.values()) for labels in partitions]
    fig,ax=plt.subplots(figsize=(10,2.6))
    ax.axis("off")
    for x,label in zip([.11,.38,.64,.89],["45 network\nflow features","64 units\nReLU · dropout 0.2","32 units\nReLU","1 unit\nSigmoid"]):
        box(ax,x,.55,label,"#e8eef5" if x==.89 else "#e2f1ef")
    for a,b in [(0.18,.28),(.49,.57),(.7,.83)]: arrow(ax,(a,.55),(b,.55))
    ax.text(.51,.17,"Shared trunk  ·  5,024 parameters",ha="center")
    ax.text(.89,.17,"Private head  ·  33",ha="center")
    ax.set_title("FedPer model  ·  5,057 trainable parameters")
    save(fig,"iomt_mlp_architecture")
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for i,ax in enumerate(axes):
        ax.axis("off")
        ax.set_title(["FedAvg","Fixed protocol groups"][i])
        for j,(name,n) in enumerate(zip(names,sizes)):
            x=[.16,.50,.84][j]
            box(ax,x,.78,f"{name}\n{n:,}", "#e2f1ef")
            dest=.5 if i==0 else (.18 if j==0 else .69)
            arrow(ax,(x,.66),(dest,.38))
        if i==0: box(ax,.5,.24,"One aggregate")
        else:
            box(ax,.18,.24,"Group 0")
            box(ax,.69,.24,"Group 1")
    save(fig,"cluster_fl_topology")
    fig,ax=plt.subplots(figsize=(7,3.5))
    x=np.arange(3)
    benign=np.array([100*sum(v["Benign"].values())/sum(sum(r.values()) for r in v.values()) for v in partitions])
    for shift,vals,label,color in [(-.18,benign,"Benign","#39877b"),(.18,100-benign,"Attack","#476b9a")]:
        bars=ax.bar(x+shift,vals,.34,color=color,label=label)
        ax.bar_label(bars,fmt="%.1f%%",padding=2)
    ax.set(xticks=x,xticklabels=names,ylim=(0,110),ylabel="Share of client records (%)")
    ax.legend(loc="upper right",ncol=2)
    save(fig,"attack_distribution")


if __name__=="__main__":
    main()
