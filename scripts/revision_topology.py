"""Draw the training budget and observed adaptive cluster topology."""
from matplotlib import pyplot as plt
from scripts.make_all_tables_figures import fig_tex, save
from src.revision_data import read_config


def topology(results):
    cfg=read_config()
    seed=cfg["seeds"][0]
    labels={"wifi":"Wi-Fi","mqtt":"MQTT","bluetooth":"Bluetooth"}
    variants=(("FedAvg",[cfg["clients"]]),("Fixed groups",cfg["fixed_groups"]),
              (f"Adaptive groups, seed {seed}",results["adaptive_groups"][str(seed)]))
    fig,axs=plt.subplots(1,3,figsize=(10.5,3.5))
    for ax,(name,groups) in zip(axs,variants):
        ax.set_title(name)
        ax.set_xlim(0,1)
        ax.set_ylim(0,1)
        ax.axis("off")
        nodes=dict(zip(cfg["clients"],(.18,.5,.82)))
        destinations=[.5] if len(groups)==1 else [(.18+.64*i/(len(groups)-1)) for i in range(len(groups))]
        for members,x in zip(groups,destinations):
            for member in members:
                node=nodes[member]
                ax.text(node,.83,labels[member],ha="center",va="center",fontsize=9,
                        bbox=dict(boxstyle="round",fc="#e2f1ef",ec="#315069"))
                ax.annotate("",xy=(x,.37),xytext=(node,.73),arrowprops=dict(arrowstyle="->",color="#345c77"))
            ax.text(x,.3,"Aggregate",ha="center",fontsize=8)
        ax.text(.5,.07,"Validation and testing use the client model",ha="center",fontsize=8)
    fig.suptitle(f"{cfg['rounds']} training rounds, {cfg['local_epochs']} local epoch per round")
    save(fig,"topology")
    return fig_tex("topology","Training and parameter exchange by aggregation policy. Raw features stay with each simulated client during federated training. FedPer additionally retains its classification head locally. Pooled training is an offline comparator.")


def benchmark_figures(config, inventory):
    from scripts.make_protocol_figures import arrow, box
    fig,ax=plt.subplots(figsize=(9,2.4))
    ax.axis("off")
    w1,w2=config["widths"]
    labels=["45 network\nflow features",f"{w1} units\nReLU and dropout",f"{w2} units\nReLU","1 unit\nSigmoid"]
    for x,label in zip((.11,.38,.64,.89),labels):
        box(ax,x,.55,label)
    for a,b in ((.18,.28),(.46,.56),(.71,.82)):
        arrow(ax,(a,.55),(b,.55))
    trunk=(45+1)*w1+(w1+1)*w2
    head=w2+1
    ax.text(.51,.18,f"Shared trunk  ({trunk:,} parameters)",ha="center")
    ax.text(.89,.18,f"Private head  ({head:,})",ha="center")
    ax.set_title(f"FedPer model  ({trunk+head:,} trainable parameters)")
    fig.savefig("figures/fig_iomt_mlp_architecture.png",dpi=300,bbox_inches="tight")
    plt.close(fig)

    clients=config["clients"]
    usage=[sum(sum(v.values()) for (c,_),v in inventory.items() if c==k) for k in clients]
    benign=[100*sum(inventory[(k,"Benign")].values())/n for k,n in zip(clients,usage)]
    x=range(len(clients))
    fig,ax=plt.subplots(figsize=(7,3.5))
    for shift,values,label,color in [(-.18,benign,"Benign","#39877b"),(.18,[100-b for b in benign],"Attack","#476b9a")]:
        bars=ax.bar([i+shift for i in x],values,.34,color=color,label=label)
        ax.bar_label(bars,fmt="%.1f%%",padding=2)
    ax.set(xticks=list(x),xticklabels=["Wi-Fi","MQTT","Bluetooth"],ylim=(0,110),ylabel="Share of client records (%)")
    ax.legend(loc="upper right",ncol=2)
    fig.tight_layout()
    fig.savefig("figures/fig_attack_distribution.png",dpi=300,bbox_inches="tight")
    plt.close(fig)
