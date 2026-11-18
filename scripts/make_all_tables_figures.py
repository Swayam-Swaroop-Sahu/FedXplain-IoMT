"""Build the quantitative appendix and figures for the validation-split study."""
import argparse
from collections import defaultdict
import json
import re
from pathlib import Path
from statistics import mean, stdev

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.revision_data import read_config


ROOT=Path("results/revision")
FIG=Path("figures")
NAMES={"local_only":"Local-only","centralized":"Centralized","fedavg":"FedAvg",
       "fedprox":"FedProx","fedper":"FedPer","fixed_cluster":"Fixed groups",
       "adaptive_cluster":"Adaptive groups","centralized_matched":"Compute-matched centralized",
       "equal_weight":"Equal client weights","class_balanced":"Class-balanced loss"}


def table(title,key,headers,rows):
    wide=len(headers)>=5
    env="table*" if wide else "table"
    width=r"\textwidth" if wide else r"\columnwidth"
    grid="\n".join([r"\begin{tabular}{l"+"r"*(len(headers)-1)+r"}\hline",
        " & ".join(headers)+r"\\\hline",
        *(" & ".join(map(str,row))+r"\\" for row in rows),
        r"\hline\end{tabular}"])
    body=r"\small"+grid if len(headers)<=3 else r"\resizebox{"+width+r"}{!}{"+grid+"}"
    return "\n".join([r"\begin{"+env+r"}[t]\centering",
        r"\caption{"+title+r"}\label{tab:"+key+r"}",body,r"\end{"+env+"}"])


def read(path):
    return json.loads(path.read_text(encoding="utf8"))


def fmt(value):
    return f"{value:.4f}" if value is not None else "--"


def summarize(values):
    return fmt(mean(values))+(r" $\pm$ " + fmt(stdev(values)) if len(values)>1 else "")


def save(fig,name):
    fig.tight_layout()
    fig.savefig(FIG/f"revision_{name}.png",dpi=300,bbox_inches="tight")
    plt.close(fig)


def fig_tex(name,caption):
    return "\n".join([r"\begin{figure*}[t]\centering",
        r"\includegraphics[width=0.93\textwidth]{revision_"+name+r".png}",
        r"\caption{"+caption+r"}\label{fig:revised_"+name+r"}",r"\end{figure*}"])


def btrecall(run):
    return run["client_metrics"]["bluetooth"]["benign_recall"]


def run_key(run):
    return (run["method"],run["mu"])


def name(key):
    return NAMES[key[0]]+(f" {key[1]:g}" if key[0]=="fedprox" else "")


def grouped_ci(root,seed,comparison,reference="fedavg"):
    """Paired empirical bootstrap using exact-feature groups for one seed."""
    data=[np.load(root/"preds"/f"{tag}_bluetooth_{seed}.npz",allow_pickle=False)
          for tag in (comparison,reference)]
    if not np.array_equal(data[0]["sample_id"],data[1]["sample_id"]):
        raise ValueError("Unpaired predictions")
    benign=data[0]["y_true"]==0
    _,group=np.unique(data[0]["group"][benign],return_inverse=True)
    n=np.bincount(group)
    diff=np.bincount(group,weights=(data[0]["logit"][benign]<0).astype(int)-(data[1]["logit"][benign]<0))
    rng=np.random.default_rng(seed)
    sample=rng.integers(0,len(n),size=(2000,len(n)))
    estimates=diff[sample].sum(axis=1)/n[sample].sum(axis=1)
    for item in data:item.close()
    return np.quantile(estimates,[.025,.975]).tolist()


def detection(root):
    grid=defaultdict(list)
    for path in sorted((root/"runs").glob("*.json")):
        run=read(path)
        grid[run_key(run)].append(run)
    cfg=read_config()
    seeds=cfg["seeds"]
    expected=len(seeds)
    assert len(grid)==14 and all(len(v)==expected for v in grid.values()), "Incomplete baseline grid"
    rows=[(name(k),summarize([x["client_mean_macro_f1"] for x in v]),summarize([x["pooled_macro_f1"] for x in v]),
           summarize([btrecall(x) for x in v]),summarize([x["optimizer_steps"] for x in v]))
          for k,v in grid.items()]
    output=[table(f"Grouped evaluation across {expected} matched seeds. Entries are means and sample standard deviations.",
                  "newmain",["Method","Client mean macro F1","Pooled macro F1","Bluetooth benign recall","Optimizer steps"],rows)]
    for seed in seeds[:2]:
        rows=[]
        for k,v in grid.items():
            r=next(x for x in v if x["seed"]==seed)
            cells=[r["client_metrics"][c][m] for m in ("macro_f1","benign_recall") for c in cfg["clients"]]
            rows.append((name(k),*[fmt(x) for x in cells],fmt(r["client_mean_macro_f1"])))
        output.append(table(f"Grouped test results for seed {seed}. BT denotes Bluetooth.",f"newseed{seed}",
                            ["Method","Wi-Fi F1","MQTT F1","BT F1","Wi-Fi benign","MQTT benign","BT benign","Client mean"],rows))
    fig,axes=plt.subplots(1,2,figsize=(10,3.6))
    ordered=sorted((k,v) for k,v in grid.items() if k[0]=="fedprox")
    x=range(len(ordered))
    for seed in seeds:
        axes[0].plot(x,[btrecall(next(x for x in v if x["seed"]==seed)) for k,v in ordered],"o-",label=str(seed))
    axes[0].set(xticks=list(x),xticklabels=[f"{k[0][1]:g}" for k in ordered],xlabel="Proximal penalty",ylabel="Bluetooth benign recall",ylim=(0,1.05))
    axes[0].legend(title="Seed",fontsize=7,ncol=2)
    for key in (("fedavg",0),("fedper",0),("fixed_cluster",0),("local_only",0),("centralized_matched",0)):
        runs=grid[key]
        for r in runs:
            assert len(r["history"])==cfg["rounds"]
        series=[[mean(h["validation_bce"].values()) for h in r["history"]] for r in runs]
        axes[1].plot(range(1,cfg["rounds"]+1),np.mean(series,axis=0),label=name(key))
    axes[1].set(xlabel="Round or matched block",ylabel="Mean validation BCE")
    axes[1].legend(fontsize=7)
    save(fig,"sweep_budget")
    output.append(fig_tex("sweep_budget","Left: test specificity for the proximal sweep. Right: validation binary cross-entropy averaged over clients and seeds."))
    compare=["fedper","fixed_cluster","adaptive_cluster","equal_weight","class_balanced"]
    changes={m:{str(s):btrecall(next(x for x in grid[(m,0)] if x["seed"]==s))
                        -btrecall(next(x for x in grid[("fedavg",0)] if x["seed"]==s)) for s in seeds} for m in compare}
    intervals={m:grouped_ci(root,seeds[0],m) for m in compare}
    return output,grid,{"paired_bluetooth_differences":changes,"seed42_feature_group_intervals":intervals}


def make_report(root):
    from scripts.revision_topology import benchmark_figures, topology
    from scripts.revision_extra_tables import calibration,shap_results,supplementary
    output,grid,results=detection(root)
    extras,probe=calibration(root)
    output.extend(extras)
    extras,shap=shap_results(root)
    output.extend(extras)
    output.extend(supplementary(root))
    cfg=read_config()
    seeds=cfg["seeds"]
    seed=seeds[0]
    split=read(root/"splits"/f"seed{seed}.json")
    inventory=defaultdict(dict)
    for record in split["counts"]:
        k=(record["client"],record["label"])
        inventory[k][record["split"]]=record["rows"]
    rows=[]
    base=next(r for r in grid[("fedavg",0)] if r["seed"]==seed)
    personalized=next(r for r in grid[("fedper",0)] if r["seed"]==seed)
    for (client,label), counts in inventory.items():
        a=base["client_metrics"][client]["per_source_label"].get(label,{})
        b=personalized["client_metrics"][client]["per_source_label"].get(label,{})
        rows.append((label.replace("_",r"\_"),client,*(counts.get(s,0) for s in ("train","validation","test")),
                    fmt(a.get("recall")),fmt(b.get("recall"))))
    output.append(table(f"Source-label supports and binary detector recall, seed {seed}.",
        "newsupport",["Label","Client","Train","Validation","Test","FedAvg recall","FedPer recall"],rows))
    benchmark_figures(cfg,inventory)
    taxonomy=[]
    for (client,label),counts in inventory.items():
        if label=="Benign":
            continue
        if "Recon" in label:
            impact="Discovery of exposed monitoring services"
        elif "Spoof" in label:
            impact="Misdirection of gateway-bound telemetry"
        elif "Connect" in label:
            impact="Less broker capacity for legitimate sessions"
        elif "Publish" in label:
            impact="Delayed monitoring topic updates"
        elif "Malformed" in label:
            impact="Application endpoint processing errors"
        elif client=="bluetooth":
            impact="Gaps in wearable observations"
        else:
            impact="Contention with monitor traffic"
        taxonomy.append((label.replace("_",r"\_"),client,f"{sum(counts.values()):,}",impact,r"\cite{dadkhah2024towards}"))
    output.append(table("Attack labels and potential service consequences in the derived subset.","attacks",
                        ["Label","Client","Records","Possible service impact","Source"],taxonomy))
    grouped=grid[("adaptive_cluster",0)]
    results["adaptive_groups"]={str(r["seed"]):r["history"][r["selected_round"]-1]["groups"] for r in grouped}
    output.append(topology(results))
    groups=[[s, " / ".join("+".join(g) for g in results["adaptive_groups"][str(s)])] for s in seeds]
    output.append(table("Adaptive group membership at each validation-selected checkpoint.","newgroups",["Seed","Client grouping"],groups))
    transfer=[read(p) for p in (root/"transfer").glob("seed*.json")]
    if transfer:
        reference=[]
        for key in ("fedavg","fedper","fixed_cluster","adaptive_cluster","local_only","centralized"):
            entries=[source[key] for r in transfer for source in r["bluetooth"].values()]
            macro=[x for source in entries for x in source["attack"].values()]
            reference.append((name((key,0)),fmt(mean(x["recall"] for x in macro)),
                              fmt(mean(x["benign_fpr"] for x in entries)),fmt(mean(x["margin"] for x in macro))))
        output.append(table("Foreign-feature transfer to the Bluetooth model. Attack metrics weight source labels equally.",
                       "newtransfer",["Target model","Attack recall","Source-mean benign FPR","Detection margin"],reference))
    holdout=defaultdict(list)
    for p in Path("results/revision_holdout/runs").glob("*.json"):
        run=read(p)
        s=run["client_metrics"]["wifi"]["per_source_label"]
        withheld=[s[k] for k in s if k.startswith("Recon")]
        holdout[run_key(run)].append(mean(r["recall"] for r in withheld))
    if holdout:
        output.append(table("Binary recall on the omitted reconnaissance labels. Each retained test label has equal weight.",
            "newholdout",["Training method","Seeds","Reconnaissance recall"],[(name(k),len(v),summarize(v)) for k,v in sorted(holdout.items())]))
    for method in ("fedavg","fedper","local_only","centralized","fixed_cluster","adaptive_cluster","equal_weight","class_balanced"):
        results[method]={metric:mean(getter(r) for r in grid[(method,0)])
                         for metric,getter in {"f1":lambda r:r["client_mean_macro_f1"],"bluetooth_specificity":btrecall}.items()}
    if probe:
        results["diagnostic"]={k:{m:mean(x["variants"][k][m] for x in probe)
                         for m in ("benign_recall","attack_recall","brier","log_loss","roc_auc")}
                         for k in probe[0]["variants"]}
        results["cka"]={k:mean(x["cka"][k] for x in probe) for k in probe[0]["cka"]}
    if shap:
        results["within_bluetooth_attack_spearman"]=mean(r["within_client"]["bluetooth_attack"]["spearman"] for r in shap)
        results["within_bluetooth_benign_spearman"]=mean(r["within_client"]["bluetooth_benign"]["spearman"] for r in shap)
        results["cross_wifi_bluetooth_attack_spearman"]=mean(r["cross_client"]["fedavg"]["wifi-bluetooth_attack"]["spearman"] for r in shap)
    sensitivity={}
    for file in (root/"sensitivity").glob("*.json"):
        r=read(file)
        if r.get("setting") in ("head_1","head_2"):
            sensitivity.setdefault(r["setting"],[]).append(r["bluetooth_specificity"])
    results["head_specificity"]={k:mean(v) for k,v in sensitivity.items()}
    results["shortcut_accuracy"]=mean(r["accuracy"] for r in read(root/"shortcut.json")["runs"])
    results["seeds"]=seeds
    results["blocked_near_signature_counts"]={str(s):read(root/"splits"/f"seed{s}.json")["rounded_signatures_across_splits"] for s in seeds}
    grouped_blocks={
        "RevisionTaxonomy":("attacks",),
        "RevisionTopology":("revised_topology","newgroups"),
        "RevisionDetection":("newmain","newseed42","newseed7","newsupport"),
        "RevisionBudget":("revised_sweep_budget",),
        "RevisionDiagnostics":("newprobe","revised_diagnostic"),
        "RevisionShap":("newshapwithin","newshapacross","revised_shap_tradeoff"),
        "RevisionSensitivity":("newsensitivity","revised_sensitivity"),
        "RevisionTransfer":("newtransfer","newholdout"),
        "RevisionScarcity":("revised_scarcity",)}
    rendered={re.search(r"\\label\{(?:fig|tab):([^}]+)\}",block)[1]:block for block in output}
    definitions=[]
    for macro,labels in grouped_blocks.items():
        blocks="\n\n".join(rendered.pop(k,"") for k in labels)
        definitions.append("\\newcommand{\\"+macro+"}{%\n"+blocks+"\n}\n")
    assert not rendered, "Unassigned tables or figures"
    return "\n".join(definitions),results



def number_sources(results):
    cfg=read_config()
    source="results/revision/paper_summary.json"
    values={
        "RevisionSeeds":(len(results["seeds"]),source),
        "RevisionRounds":(cfg["rounds"],"config/revision.yml"),
        "RevisionEpochs":(cfg["local_epochs"],"config/revision.yml"),
        "RevisionBatch":(cfg["batch_size"],"config/revision.yml"),
        "RevisionLr":(cfg["lr"],"config/revision.yml"),
        "RevisionWidthA":(cfg["widths"][0],"config/revision.yml"),
        "RevisionWidthB":(cfg["widths"][1],"config/revision.yml"),
        "RevisionDropout":(cfg["dropout"],"config/revision.yml"),
        "RevisionBg":(cfg["shap_background"],"config/revision.yml"),
        "RevisionSample":(cfg["shap_samples"],"config/revision.yml"),
        "RevisionBootstrap":(cfg["shap_bootstrap"],"config/revision.yml"),
    }
    for alias,key in [("Avg","fedavg"),("Per","fedper"),("Local","local_only"),("Fixed","fixed_cluster"),("Adaptive","adaptive_cluster"),("Equal","equal_weight"),("Balanced","class_balanced")]:
        values[f"Revision{alias}F"]=fmt(results[key]["f1"]),source
        values[f"Revision{alias}B"]=fmt(results[key]["bluetooth_specificity"]),source
    values["RevisionShortcut"]=fmt(results["shortcut_accuracy"]*100),source
    for suffix,key in [("One","head_1"),("Two","head_2")]:
        values["RevisionHeadDepth"+suffix]=fmt(results["head_specificity"][key]),source
    lower,upper=results["seed42_feature_group_intervals"]["fedper"]
    values["RevisionPerLo"]=fmt(lower),source
    values["RevisionPerHi"]=fmt(upper),source
    if "diagnostic" in results:
        for alias,key in [("ProbeAvg","fedavg"),("Platt","platt"),("Temperature","temperature"),("Threshold","threshold"),
                          ("Head","frozen_trunk"),("Tune","fine_tune")]:
            for suffix,metric in [("B","benign_recall"),("A","attack_recall"),("Brier","brier")]:
                values[f"Revision{alias}{suffix}"]=fmt(results["diagnostic"][key][metric]),source
        values["RevisionCka"]=fmt(results["cka"]["fedavg-fedper"]),source
    if "within_bluetooth_attack_spearman" in results:
        values["RevisionWithinAttack"]=fmt(results["within_bluetooth_attack_spearman"]),source
        values["RevisionWithinBenign"]=fmt(results["within_bluetooth_benign_spearman"]),source
        values["RevisionAcrossAttack"]=fmt(results["cross_wifi_bluetooth_attack_spearman"]),source
    return values

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check",action="store_true")
    args=parser.parse_args()
    tex,results=make_report(ROOT)
    sources=number_sources(results)
    numbers="".join(f"\\newcommand{{\\{key}}}{{{val}}}\n" for key,(val,_) in sources.items())
    numbers_path=Path("revision_numbers.tex")
    path=Path("revision_tables.tex")
    payload=json.dumps(results,indent=2,allow_nan=False)+"\n"
    report=ROOT/"paper_summary.json"
    readme=Path("README.md")
    current=readme.read_text(encoding="utf8")
    begin,end="<!-- revision-summary:start -->","<!-- revision-summary:end -->"
    before,rest=current.split(begin,1)
    _,after=rest.split(end,1)
    lines=["","| Method | Client-mean macro F1 | Bluetooth benign recall |","|---|---:|---:|"]
    for key in ("fedavg","fedper","fixed_cluster","adaptive_cluster","equal_weight","class_balanced","local_only","centralized"):
        r=results[key]
        lines.append(f"| {NAMES[key]} | {fmt(r['f1'])} | {fmt(r['bluetooth_specificity'])} |")
    rendered=before+begin+"\n".join(["",*lines,""])+end+after
    if args.check:
        assert current==rendered, "README numerical table differs"
        assert numbers_path.read_text(encoding="utf8")==numbers, "Numerical macros differ"
        assert path.read_text(encoding="utf8")==tex, "Tables differ from generated results"
        assert report.read_text(encoding="utf8")==payload, "Paper summary differs from generated results"
    else:
        readme.write_text(rendered,encoding="utf8")
        numbers_path.write_text(numbers,encoding="utf8")
        path.write_text(tex,encoding="utf8")
        claims="claim,value,source,script\n"+"".join(f"{key},{v},{source},scripts/make_all_tables_figures.py\n" for key,(v,source) in sources.items())
        Path("docs/claim_to_result_map.csv").write_text(claims,encoding="utf8")
        report.write_text(payload,encoding="utf8")


if __name__=="__main__":
    main()
