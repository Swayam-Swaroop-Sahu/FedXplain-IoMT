"""Publish inventories and audit protocol information in non-applicable fields."""
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix
from sklearn.tree import DecisionTreeClassifier

from src.revision_data import assign_split, read_config, read_rows
from src.run_revision import write_json


def main():
    config=read_config()
    rows,features=read_rows(config["clients"])
    fields=json.loads(Path("results_v1_archive/feature_dictionary.json").read_text())
    inventory=[]
    outputs=[]
    # Only documented inapplicable fields that are empirically absent in Bluetooth.
    restricted=[name for name,info in fields.items()
                if info["protocols"]["bluetooth"]["all_zero_in_subset"]
                and "inapplicable" in info["protocols"]["bluetooth"]["mapping"]]
    for seed in config["seeds"]:
        split=assign_split(rows,seed,config["validation_fraction"],config["test_fraction"])
        training=split[split["split"].eq("train")]
        testing=split[split["split"].eq("test")]
        tree=DecisionTreeClassifier(max_depth=3,class_weight="balanced",random_state=seed)
        tree.fit(training[restricted].ne(0),training["client"])
        predicted=tree.predict(testing[restricted].ne(0))
        outputs.append({"seed":seed,"accuracy":accuracy_score(testing["client"],predicted),
            "balanced_accuracy":balanced_accuracy_score(testing["client"],predicted),
            "client_order":config["clients"],
            "confusion":confusion_matrix(testing["client"],predicted,labels=config["clients"]).tolist()})
        table=split.groupby(["client","split","label"]).size().rename("records").reset_index()
        inventory.append(table.assign(seed=seed))
    pd.concat(inventory,ignore_index=True).to_csv("docs/label_inventory.csv",index=False)
    write_json(Path("results/revision/shortcut.json"),{"features":restricted,"depth":3,"runs":outputs})
    result=["# Study feature dictionary","",
        "Feature order comes from the CSV headers. Missing, NaN and infinite inputs are replaced with zero before scaling. Exact units or extractor formulas require the source implementation.","",
        "| Feature | Meaning | Bluetooth handling | All zero in derived subset |",
        "|---|---|---|---|"]
    for name,feature in fields.items():
        bt=feature["protocols"]["bluetooth"]["mapping"]
        zeros=", ".join(k for k,v in feature["protocols"].items() if v["all_zero_in_subset"])
        result.append(f"| {name} | {feature['description']} | {bt} | {zeros or 'None'} |")
    Path("docs/feature_dictionary.md").write_text("\n".join(result)+"\n",encoding="utf8")


if __name__=="__main__":
    main()
