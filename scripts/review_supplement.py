"""Build reviewer evidence tables from derived CSVs and saved results.

Run from the repository root with python -m scripts.review_supplement.
"""
import hashlib
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

from src.config import PROTOCOLS, SEEDS, SHAP_ATTACK_SAMPLE_SIZE, SHAP_BACKGROUND_SIZE, TEST_SIZE
from src.utils import _is_drop_column


def write_table(archive, name, rows):
    archive.writestr(name, pd.DataFrame(rows).to_csv(index=False, lineterminator="\n"))


def main():
    study = json.loads(Path("results/study_results.json").read_text())
    generalization = json.loads(Path("results/generalization_results.json").read_text())
    with ZipFile("results/review_supplement.zip", "w", ZIP_DEFLATED) as archive:
        overlaps, inventory, schema, confusion = [], [], [], []
        for client in PROTOCOLS:
            path = Path("data/study") / f"{client}_study.csv"
            frame = pd.read_csv(path)
            labels = frame["label"].str.strip()
            target = labels.str.lower().ne("benign").to_numpy()
            features = frame[[c for c in frame if not _is_drop_column(c)]].select_dtypes("number")
            values = np.nan_to_num(features.to_numpy(dtype=np.float32), nan=0, posinf=0, neginf=0)
            # np.void equality compares the complete pre-scaling byte string.
            values = np.ascontiguousarray(values)
            values[values == 0] = 0  # normalize signed zero
            keys = values.view(np.dtype((np.void, values.dtype.itemsize * values.shape[1]))).ravel()
            for j, name in enumerate(features):
                schema.append(dict(client=client, index=j, feature=name,
                                   nonzero_rows=int(np.count_nonzero(values[:, j])),
                                   distinct_values=len(np.unique(values[:, j]))))
            manifest = pd.DataFrame({"csv_row_zero_based": np.arange(len(frame)), "label": labels})
            for seed in SEEDS:
                train, test = train_test_split(np.arange(len(frame)), test_size=TEST_SIZE,
                                               stratify=target, random_state=seed)
                matches = np.isin(keys[test], keys[train])
                overlaps.append(dict(client=client, seed=seed, n_test=len(test),
                                     identical_to_training=int(matches.sum()),
                                     csv_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
                split = np.full(len(frame), "train", dtype=object)
                split[test] = "test"
                manifest[f"split_{seed}"] = split
                rng = torch.Generator().manual_seed(seed)
                for group, candidates, n in [("background", train[~target[train]], SHAP_BACKGROUND_SIZE),
                                             ("explained", test[target[test]], SHAP_ATTACK_SAMPLE_SIZE)]:
                    selected = candidates[torch.randperm(len(candidates), generator=rng)[:n].numpy()]
                    manifest.loc[selected, f"shap_{seed}"] = group
                for label in sorted(labels.unique()):
                    inventory.append(dict(client=client, seed=seed, label=label,
                                          train=int((labels.iloc[train] == label).sum()),
                                          test=int((labels.iloc[test] == label).sum())))
            write_table(archive, f"{client}_row_manifest.csv", manifest)
        write_table(archive, "identical_feature_rows.csv", overlaps)
        write_table(archive, "label_visibility_and_support.csv", inventory)
        write_table(archive, "feature_order_and_variation.csv", schema)
        for run in [*study["runs"], *study["local_only"]]:
            for client, metrics in run["client_metrics"].items():
                confusion.append(dict(method=run.get("aggregator", "local_only"), mu=run.get("mu"),
                                      seed=run["seed"], client=client, **metrics))
        for seed, run in study["centralized_by_seed"].items():
            for client, metrics in run["client_metrics"].items():
                confusion.append(dict(method="centralized", seed=int(seed), client=client, **metrics))
        write_table(archive, "detection.csv", confusion)
        transfers = []
        for target, experiment in generalization["zero_day"]["targets"].items():
            for run in experiment["per_seed"]:
                for method, attacks in run["unseen_attack_recall"].items():
                    for label, recall in attacks.items():
                        source = experiment["unseen_attacks"][label]
                        flag = run["foreign_benign_flag_rate"][method][source["source"]]
                        transfers.append(dict(target=target, seed=run["seed"], method=method, label=label,
                                              source=source["source"], n_attack=source["n_test"][str(run["seed"])],
                                              attack_recall=recall, benign_flag_rate=flag, margin=recall-flag))
        write_table(archive, "transfer.csv", transfers)
        scarcity = generalization["data_scarcity"]["targets"]
        write_table(archive, "adaptation.csv", [dict(r, target=t) for t, e in scarcity.items() for r in e["runs"]])
        write_table(archive, "adaptation_by_budget.csv", [dict(r, target=t, method=m) for t,e in scarcity.items()
                    for m, rows in e["summary"].items() for r in rows])
        archive.writestr("adaptation_protocol.json", json.dumps({t:e["config"] for t,e in scarcity.items()},indent=2))


if __name__ == "__main__":
    main()
