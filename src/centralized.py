"""Centralized baseline: train on all three protocols combined."""

import os
import sys

import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Allow running as `python src/centralized.py` from repo root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import set_seed, load_and_preprocess, get_input_dim
from src.model import IoMTMLP


def main() -> None:
    set_seed(42)

    protocols = ["wifi", "mqtt", "bluetooth"]
    data_dir = "data/poc"

    # ------------------------------------------------------------------
    # 1. Load & concatenate all protocols
    # ------------------------------------------------------------------
    X_trains, X_tests, y_trains, y_tests = [], [], [], []
    for proto in protocols:
        Xtr, Xte, ytr, yte = load_and_preprocess(proto, data_dir=data_dir)
        X_trains.append(Xtr)
        X_tests.append(Xte)
        y_trains.append(ytr)
        y_tests.append(yte)

    X_train = torch.cat(X_trains, dim=0)
    X_test = torch.cat(X_tests, dim=0)
    y_train = torch.cat(y_trains, dim=0)
    y_test = torch.cat(y_tests, dim=0)

    input_dim = X_train.shape[1]
    print(f"Combined dataset  -> train: {X_train.shape[0]}, test: {X_test.shape[0]}, "
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
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCELoss()

    # ------------------------------------------------------------------
    # 3. Training loop (10 epochs, batch size 64)
    # ------------------------------------------------------------------
    batch_size = 64
    dataset = torch.utils.data.TensorDataset(X_train, y_train)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    print("\n--- Training ---")
    for epoch in range(1, 11):
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

        if epoch % 2 == 0:
            print(f"  Epoch {epoch:>2d}/10  |  Loss: {epoch_loss:.4f}")

    # ------------------------------------------------------------------
    # 4. Evaluation
    # ------------------------------------------------------------------
    model.eval()
    with torch.no_grad():
        y_prob = model(X_test)
        y_pred = (y_prob >= 0.5).float().numpy().flatten()
    y_true = y_test.numpy().flatten()

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    print("\n--- Test-set evaluation ---")
    print(f"{'Metric':<12} {'Value':>8}")
    print("-" * 22)
    print(f"{'Accuracy':<12} {acc:>8.4f}")
    print(f"{'Precision':<12} {prec:>8.4f}")
    print(f"{'Recall':<12} {rec:>8.4f}")
    print(f"{'F1-score':<12} {f1:>8.4f}")

    if f1 < 0.70:
        print(f"\n[WARNING] F1 ({f1:.4f}) is below 0.70. Printing class distributions:")
        print(f"  Train -> Benign: {n_benign_train}, Attack: {n_attack_train}")
        print(f"  Test  -> Benign: {n_benign_test}, Attack: {n_attack_test}")

    # ------------------------------------------------------------------
    # 5. Save model
    # ------------------------------------------------------------------
    os.makedirs("models", exist_ok=True)
    save_path = "models/centralized_baseline.pth"
    torch.save(model.state_dict(), save_path)
    print(f"\nCentralized baseline complete. Model saved to {save_path}")


if __name__ == "__main__":
    main()
