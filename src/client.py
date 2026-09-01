"""Flower NumPyClient wrapper for per-protocol IoMT data."""

import flwr as fl
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from src.config import (
    BATCH_SIZE,
    DATA_DIR,
    LEARNING_RATE,
    N_LOCAL_EPOCHS,
    PROTOCOL_MAP,
)
from src.model import IoMTMLP
from src.utils import get_input_dim, load_and_preprocess, set_seed


class IoMTFlowerClient(fl.client.NumPyClient):
    """A Flower client that trains on a single IoMT protocol's data."""

    def __init__(
        self,
        cid: int,
        protocol: str | None = None,
        data_dir: str = DATA_DIR,
        seed: int = 42,
    ) -> None:
        super().__init__()
        if protocol is None:
            protocol = PROTOCOL_MAP[cid]
        self.protocol = protocol
        self.cid = cid
        self.data_dir = data_dir
        self.seed = seed

        set_seed(seed)
        self.X_train, self.X_test, self.y_train, self.y_test = load_and_preprocess(
            self.protocol, data_dir=self.data_dir, random_state=self.seed
        )
        input_dim = self.X_train.shape[1]
        self.model = IoMTMLP(input_dim)

    # ------------------------------------------------------------------
    # Flower NumPyClient interface
    # ------------------------------------------------------------------
    def get_parameters(self, config: dict | None = None) -> list[np.ndarray]:
        return self.model.get_weights()

    def set_parameters(self, parameters: list[np.ndarray]) -> None:
        self.model.set_weights(parameters)

    def fit(
        self, parameters: list[np.ndarray], config: dict | None = None
    ) -> tuple[list[np.ndarray], int, dict]:
        self.set_parameters(parameters)

        mu = float(config.get("mu", 0.0)) if config else 0.0
        epochs = int(config.get("epochs", N_LOCAL_EPOCHS)) if config else N_LOCAL_EPOCHS
        batch_size = int(config.get("batch_size", BATCH_SIZE)) if config else BATCH_SIZE
        lr = float(config.get("lr", LEARNING_RATE)) if config else LEARNING_RATE

        if mu > 0.0:
            global_params = [p.clone().detach() for p in self.model.parameters()]

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.BCELoss()
        dataset = torch.utils.data.TensorDataset(self.X_train, self.y_train)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        self.model.train()
        final_loss = 0.0
        for _epoch in range(epochs):
            epoch_loss = 0.0
            for X_batch, y_batch in loader:
                optimizer.zero_grad()
                preds = self.model(X_batch)
                loss = criterion(preds, y_batch)
                if mu > 0.0:
                    prox_term = sum(
                        torch.sum((p - g_p) ** 2)
                        for p, g_p in zip(self.model.parameters(), global_params)
                    )
                    loss = loss + (mu / 2.0) * prox_term
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * X_batch.size(0)
            final_loss = epoch_loss / len(dataset)

        print(
            f"  Client {self.protocol:<10s} - Local training complete. "
            f"Loss: {final_loss:.4f}"
        )
        return (
            self.get_parameters({}),
            len(self.X_train),
            {"loss": float(final_loss)},
        )

    def evaluate(
        self, parameters: list[np.ndarray], config: dict | None = None
    ) -> tuple[float, int, dict]:
        self.set_parameters(parameters)

        self.model.eval()
        criterion = nn.BCELoss()
        with torch.no_grad():
            y_prob = self.model(self.X_test)
            loss = criterion(y_prob, self.y_test).item()
            y_pred = (y_prob >= 0.5).float().numpy().flatten()

        y_true = self.y_test.numpy().flatten()

        # Class-imbalance-aware metrics
        cm = confusion_matrix(y_true, y_pred)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
        else:
            # Handle edge case where only one class exists in batch
            tn = int(np.sum((y_true == 0) & (y_pred == 0)))
            fp = int(np.sum((y_true == 0) & (y_pred == 1)))
            fn = int(np.sum((y_true == 1) & (y_pred == 0)))
            tp = int(np.sum((y_true == 1) & (y_pred == 1)))

        benign_prec = float(tn / (tn + fn)) if (tn + fn) > 0 else 0.0
        benign_rec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
        attack_prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        attack_rec = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0

        f1_binary = float(f1_score(y_true, y_pred, zero_division=0))
        f1_macro = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
        f1_weighted = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
        acc = float(accuracy_score(y_true, y_pred))

        metrics = {
            "f1": f1_binary,
            "f1_binary": f1_binary,
            "f1_macro": f1_macro,
            "f1_weighted": f1_weighted,
            "accuracy": acc,
            "benign_precision": benign_prec,
            "benign_recall": benign_rec,
            "attack_precision": attack_prec,
            "attack_recall": attack_rec,
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
            "n_benign": int(np.sum(y_true == 0)),
            "n_attack": int(np.sum(y_true == 1)),
        }

        return (float(loss), len(self.X_test), metrics)
