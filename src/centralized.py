"""Centralized baseline: train on all three protocols combined."""

import os
import sys

import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

# Allow running as `python src/centralized.py` from repo root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import (
    BATCH_SIZE,
    CENTRALIZED_EPOCHS,
    DATA_DIR,
    LEARNING_RATE,
    MODELS_DIR,
    PROTOCOLS,
)
from src.model import IoMTMLP
from src.utils import get_input_dim, load_and_preprocess, set_seed


def main(data_dir: str = DATA_DIR, seed: int = 42) -> dict:
    set_seed(seed)

    # ------------------------------------------------------------------
    # 1. Load & concatenate all protocols
    # ------------------------------------------------------------------
    X_trains, X_tests, y_trains, y_tests = [], [], [], []
    for proto in PROTOCOLS:
        Xtr, Xte, ytr, yte = load_and_preprocess(proto, data_dir=data_dir, random_state=seed)
        X_trains.append(Xtr)
        X_tests.append(Xte)
        y_trains.append(ytr)
        y_tests.append(yte)

    X_train = torch.cat(X_trains, dim=0)
    X_test = torch.cat(X_tests, dim=0)
    y_train = torch.cat(y_trains, dim=0)
    y_test = torch.cat(y_tests, dim=0)

    input_dim = X_train.shape[1]

    # Guardrail: verify all protocols produce the same feature count
    for proto in PROTOCOLS:
        proto_dim = get_input_dim(proto, data_dir=data_dir)
        assert proto_dim == input_dim, (
            f"input_dim mismatch: '{proto}' has {proto_dim} features but "
            f"expected {input_dim}. Preprocessing may have changed."
        )

    print(f"Combined dataset ({data_dir}) -> train: {X_train.shape[0]}, test: {X_test.shape[0]}, "
          f"features: {input_dim}")

    # Class distribution
    n_benign_train = int((y_train == 0).sum())
    n_attack_train = int((y_train == 1).sum())
    n_benign_test = int((y_test == 0).sum())
    n_attack_test = int((y_test == 1).sum())
    print(f"Train distribution -> Benign: {n_benign_train}, Attack: {n_attack_train}")
    print(f"Test  distribution -> Benign: {n_benign_test}, Attack: {n_attack_test}")

    # ------------------------------------------------------------------
    # 2. Model, optimizer, loss
    # ------------------------------------------------------------------
    model = IoMTMLP(input_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.BCELoss()

    # ------------------------------------------------------------------
    # 3. Training loop
    # ------------------------------------------------------------------
    dataset = torch.utils.data.TensorDataset(X_train, y_train)
    loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    print(f"\n--- Training ({CENTRALIZED_EPOCHS} epochs, batch {BATCH_SIZE}) ---")
    for epoch in range(1, CENTRALIZED_EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        for X_batch, y_batch in loader:
            optimizer.zero_grad()
            preds = model(X_batch)
            loss = criterion(preds, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * X_batch.size(0)
        epoch_loss /= len(dataset)

        if epoch % 2 == 0 or epoch == CENTRALIZED_EPOCHS:
            print(f"  Epoch {epoch:>2d}/{CENTRALIZED_EPOCHS}  |  Loss: {epoch_loss:.4f}")

    # ------------------------------------------------------------------
    # 4. Evaluation
    # ------------------------------------------------------------------
    model.eval()
    with torch.no_grad():
        y_prob = model(X_test)
        y_pred = (y_prob >= 0.5).float().numpy().flatten()
    y_true = y_test.numpy().flatten()

    acc = float(accuracy_score(y_true, y_pred))
    f1_bin = float(f1_score(y_true, y_pred, zero_division=0))
    f1_mac = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    f1_wt = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)
    benign_prec = float(tn / (tn + fn)) if (tn + fn) > 0 else 0.0
    benign_rec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    attack_prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    attack_rec = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0

    print("\n--- Centralized Test-set Evaluation ---")
    print(f"{'Metric':<20} {'Value':>10}")
    print("-" * 32)
    print(f"{'Accuracy':<20} {acc:>10.4f}")
    print(f"{'Binary F1 (Attack)':<20} {f1_bin:>10.4f}")
    print(f"{'Macro F1':<20} {f1_mac:>10.4f}")
    print(f"{'Weighted F1':<20} {f1_wt:>10.4f}")
    print(f"{'Benign Precision':<20} {benign_prec:>10.4f}")
    print(f"{'Benign Recall':<20} {benign_rec:>10.4f}")
    print(f"{'Attack Precision':<20} {attack_prec:>10.4f}")
    print(f"{'Attack Recall':<20} {attack_rec:>10.4f}")

    # ------------------------------------------------------------------
    # 5. Save model
    # ------------------------------------------------------------------
    os.makedirs(MODELS_DIR, exist_ok=True)
    save_path = os.path.join(MODELS_DIR, "centralized_baseline.pth")
    torch.save(model.state_dict(), save_path)
    print(f"\nCentralized baseline complete. Model saved to {save_path}")

    return {
        "accuracy": acc,
        "f1_binary": f1_bin,
        "f1_macro": f1_mac,
        "f1_weighted": f1_wt,
        "benign_precision": benign_prec,
        "benign_recall": benign_rec,
        "attack_precision": attack_prec,
        "attack_recall": attack_rec,
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


if __name__ == "__main__":
    main()
