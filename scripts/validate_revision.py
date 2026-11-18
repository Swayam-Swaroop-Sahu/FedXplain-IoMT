"""Verify split, checkpoint metadata, step and prediction invariants from revision outputs."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.revision_metrics import metrics
from src.revision_data import read_config
from src.run_revision import baseline_grid


def validate(root,split_dir,seeds):
    cfg=read_config()
    checks=0
    for seed in seeds:
        manifest=pd.read_csv(split_dir/f"manifest_{seed}.csv",keep_default_na=False)
        assert manifest["record_id"].is_unique
        assert manifest.groupby("group")["split"].nunique().max()==1
        test=manifest.loc[manifest["split"].eq("test")]
        for method,mu in baseline_grid(cfg):
            key=f"fedprox_{mu}" if method=="fedprox" else method
            run=json.loads((root/"runs"/f"{key}_{seed}.json").read_text())
            assert run["seed"]==seed and run["config"]==cfg
            val_scores=[np.mean(list(r["validation_bce"].values())) for r in run["history"]]
            assert run["selected_round"]==int(np.argmin(val_scores))+1
            assert run["optimizer_steps"]==sum(s["steps"] for h in run["history"] for s in h["train"].values())
            y,z=[],[]
            scores=[]
            for k in cfg["clients"]:
                with np.load(root/"preds"/f"{key}_{k}_{seed}.npz",allow_pickle=False) as p:
                    ids=test.loc[test["client"].eq(k),"record_id"].to_numpy(dtype=str)
                    assert np.array_equal(ids,p["sample_id"])
                    measured=metrics(p["y_true"],p["logit"],bins=cfg["reliability_bins"])
                    agreed=run["client_metrics"][k]
                    for metric in ("tn","fp","fn","tp","macro_f1","roc_auc","pr_auc","brier","ece"):
                        assert np.isclose(measured[metric],agreed[metric],rtol=0,atol=1e-12),(key,k,seed,metric)
                    y.append(p["y_true"]); z.append(p["logit"]);scores.append(measured["macro_f1"])
                    checks+=1
            assert np.isclose(metrics(np.concatenate(y),np.concatenate(z))["macro_f1"],run["pooled_macro_f1"])
            assert np.isclose(np.mean(scores),run["client_mean_macro_f1"])
        for k in cfg["clients"]:
            with np.load(root/"preds"/f"fedavg_{k}_{seed}.npz") as a:
                with np.load(root/"preds"/f"fedprox_0_{k}_{seed}.npz") as b:
                    np.testing.assert_array_equal(a["logit"],b["logit"])
        left=json.loads((root/"runs"/f"fedavg_{seed}.json").read_text())
        right=json.loads((root/"runs"/f"centralized_matched_{seed}.json").read_text())
        assert left["optimizer_steps"]==right["optimizer_steps"]
    print(f"{root}: checked {checks} client prediction files, validation-only selection, zero-penalty parity and matched updates")


if __name__=="__main__":
    c=read_config()
    validate(Path("results/revision"),Path("splits/revision"),c["seeds"])
    validate(Path("results/revision_holdout"),Path("splits/revision_holdout"),c["seeds"][:2])
