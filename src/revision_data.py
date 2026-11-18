"""Feature-table ingestion with split provenance and train-only transforms.

The optional provenance manifest joins on client and zero-based row. Source
metadata are supplied by the extractor, never inferred from traffic.
"""
import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
import yaml


def read_config(path="config/revision.yml"):
    return yaml.safe_load(Path(path).read_text())


def read_rows(clients, root="data/study", provenance=None):
    frames, features = [], None
    for client in clients:
        path = Path(root) / f"{client}_study.csv"
        frame = pd.read_csv(path)
        columns = frame.columns.drop("label").tolist()
        if len(columns) != 45 or (features is not None and columns != features):
            raise ValueError(f"Wrong 45-column feature schema in {path}")
        features = columns
        frame[features] = frame[features].astype("float32").replace([np.inf, -np.inf], np.nan).fillna(0)
        frame[features] = frame[features] + np.float32(0)  # normalize signed zero
        frame["client"] = client
        frame["row"] = np.arange(len(frame))
        frame["record_id"] = client + ":" + frame["row"].astype(str)
        frame["feature_file"] = path.name
        frame["label"] = frame["label"].str.strip()
        frames.append(frame)
    rows = pd.concat(frames, ignore_index=True)
    rows["feature_group"] = pd.util.hash_pandas_object(rows[features], index=False).map(lambda h: f"{h:016x}")
    rows["source_file"] = ""
    rows["group"] = rows["feature_group"]
    if provenance:
        supplied = pd.read_csv(provenance, keep_default_na=False)
        columns = ["client", "row", "source_file", "group", "official_split"]
        rows = rows.drop(columns=["source_file","group"]).merge(
            supplied[columns], on=["client","row"], how="left", validate="one_to_one")
        if rows[columns].isna().any().any() or rows["group"].eq("").any():
            raise ValueError("Every feature row requires an extractor-provided group")
        # Reject conflicting atomic groups rather than silently breaking provenance.
        if rows.groupby("feature_group")["group"].nunique().max() > 1:
            raise ValueError("Manifest must coalesce source groups joined by identical features")
    rows["y"] = rows["label"].str.casefold().ne("benign").astype(int)
    return rows, features


def assign_split(rows, seed, validation_fraction, test_fraction):
    rows = rows.copy()
    official = rows.get("official_split")
    if official is not None and official.ne("").any():
        if not official.isin(["train", "test"]).all():
            raise ValueError("Official split must specify train or test on every row")
        trainval = np.flatnonzero(official.eq("train"))
        test = np.flatnonzero(official.eq("test"))
        relative_val = validation_fraction
    else:
        outer = GroupShuffleSplit(n_splits=1, test_size=test_fraction, random_state=seed)
        trainval, test = next(outer.split(rows, groups=rows["group"]))
        relative_val = validation_fraction / (1-test_fraction)
    inner = GroupShuffleSplit(n_splits=1, test_size=relative_val, random_state=seed+1)
    train, val = next(inner.split(trainval, groups=rows.iloc[trainval]["group"]))
    labels = np.full(len(rows), "test", dtype=object)
    labels[trainval[train]], labels[trainval[val]] = "train", "validation"
    rows["split"] = labels
    if rows.groupby("group")["split"].nunique().max() != 1:
        raise ValueError("Split crosses a source or exact-feature group")
    return rows


def prepare_clients(rows, features, held_out=()):
    """Keep original IDs and fit each scaler on its eligible training records."""
    clients = {}
    for client, block in rows.groupby("client", sort=False):
        train = block["split"].eq("train") & ~block["label"].isin(held_out)
        scaler = StandardScaler().fit(block.loc[train, features])
        clients[client] = {}
        for split in ("train","validation","test"):
            mask = train if split == "train" else block["split"].eq(split)
            if split == "validation":
                mask &= ~block["label"].isin(held_out)
            chosen = block.loc[mask]
            clients[client][split] = {
                "x": scaler.transform(chosen[features]).astype(np.float32),
                "y": chosen["y"].to_numpy(),
                "id": chosen["record_id"].to_numpy(dtype=str),
                "label": chosen["label"].to_numpy(dtype=str),
                "group": chosen["group"].to_numpy(dtype=str),
            }
    return clients


def split_report(rows, features):
    # A collision in a four-significant-digit signature flags possible near matches.
    # It is a screen, not a geometric nearest-neighbor guarantee.
    x = rows[features].to_numpy(dtype=float)
    scale = np.where(x == 0, 1, 10 ** np.floor(np.log10(np.maximum(np.abs(x), 1e-45))))
    signature = pd.util.hash_pandas_object(pd.DataFrame(np.round(x/scale, 3)*scale), index=False)
    sig = pd.Series(signature.to_numpy(), index=rows.index)
    counts = rows.groupby(["client","split","label"]).size().rename("rows").reset_index()
    return {"counts":counts.to_dict(orient="records"),
            "exact_across_splits": int(rows.groupby("feature_group")["split"].nunique().gt(1).sum()),
            "rounded_signatures_across_splits": int(rows.groupby(sig)["split"].nunique().gt(1).sum())}


def write_manifest(rows, root, seed, expected=None):
    path = root / f"manifest_{seed}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = ["record_id","feature_file","source_file","group","client","label","split"]
    contents = rows[cols].to_csv(index=False,lineterminator="\r\n").encode("utf8")
    metadata = {"file":path.name,"sha256":hashlib.sha256(contents).hexdigest()}
    if expected is not None and metadata != expected:
        raise ValueError(f"Split manifest digest differs for seed {seed}. Use a new output directory.")
    path.write_bytes(contents)
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/study")
    parser.add_argument("--provenance")
    args = parser.parse_args()
    cfg = read_config()
    rows, features = read_rows(cfg["clients"], args.data, args.provenance)
    for seed in cfg["seeds"]:
        split = assign_split(rows,seed,cfg["validation_fraction"],cfg["test_fraction"])
        print(write_manifest(split,Path("splits"),seed))


if __name__ == "__main__":
    main()
