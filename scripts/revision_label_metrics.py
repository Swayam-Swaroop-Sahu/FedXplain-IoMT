"""Evaluate each attack label against benign examples using binary predictions."""
import csv
from pathlib import Path

import numpy as np


def main():
    fields=["method","client","seed","label","attack_support","benign_support",
            "attack_as_positive_recall","attack_vs_benign_precision"]
    with open("results/revision/per_label_metrics.csv","w",newline="",encoding="utf8") as stream:
        writer=csv.DictWriter(stream,fieldnames=fields)
        writer.writeheader()
        for path in sorted(Path("results/revision/preds").glob("*.npz")):
            method,client,seed=path.stem.rsplit("_",2)
            with np.load(path,allow_pickle=False) as data:
                y,label,pred=data["y_true"],data["label"],data["logit"]>=0
                benign=y==0
                false_positive=int(pred[benign].sum())
                for name in np.unique(label[~benign]):
                    mask=label==name
                    count=int(mask.sum())
                    positive=int(pred[mask].sum())
                    writer.writerow(dict(method=method,client=client,seed=seed,label=name,attack_support=count,
                        benign_support=int(benign.sum()),attack_as_positive_recall=positive/count,
                        attack_vs_benign_precision=(positive/(positive+false_positive)
                                                    if positive+false_positive else None)))


if __name__=="__main__":
    main()
