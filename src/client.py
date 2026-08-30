"""Flower NumPyClient wrapper for per-protocol IoMT data."""

import flwr as fl
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score

from src.utils import set_seed, load_and_preprocess, get_input_dim
from src.model import IoMTMLP

PROTOCOL_MAP = {0: "wifi", 1: "mqtt", 2: "bluetooth"}


class IoMTFlowerClient(fl.client.NumPyClient):
    """A Flower client that trains on a single IoMT protocol's data."""

    def __init__(self, cid: int, protocol: str | None = None) -> None:
        super().__init__()
        if protocol is None:
            protocol = PROTOCOL_MAP[cid]
        self.protocol = protocol
        self.cid = cid

        set_seed(42)
        self.X_train, self.X_test, self.y_train, self.y_test = load_and_preprocess(
            self.protocol
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

        # Local training: 2 epochs, batch 64, Adam lr=0.001, BCELoss
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        criterion = nn.BCELoss()
        dataset = torch.utils.data.TensorDataset(self.X_train, self.y_train)
        loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)

        self.model.train()
        final_loss = 0.0
        for _epoch in range(2):
            epoch_loss = 0.0
            for X_batch, y_batch in loader:
                optimizer.zero_grad()
                preds = self.model(X_batch)
                loss = criterion(preds, y_batch)
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
        f1 = f1_score(y_true, y_pred, zero_division=0)

        return (float(loss), len(self.X_test), {"f1": float(f1)})
