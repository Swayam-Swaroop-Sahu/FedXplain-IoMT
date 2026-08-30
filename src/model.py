"""IoMT MLP model for binary attack detection."""

from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn


class IoMTMLP(nn.Module):
    """Simple 3-layer MLP: input -> 64 -> 32 -> 1 (sigmoid)."""

    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def get_weights(self) -> list[np.ndarray]:
        """Return model parameters as a list of NumPy arrays."""
        return [param.cpu().numpy() for param in self.state_dict().values()]

    def set_weights(self, weights: list[np.ndarray]) -> None:
        """Load model parameters from a list of NumPy arrays."""
        state_dict = self.state_dict()
        keys = list(state_dict.keys())
        new_state = OrderedDict()
        for key, w in zip(keys, weights):
            new_state[key] = torch.from_numpy(w)
        self.load_state_dict(new_state, strict=True)
