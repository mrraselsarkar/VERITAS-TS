"""A second, independent judge: a small 1-D convolutional network.

The main oracle (MultiRocket-Hydra + QUANT) is built from random-kernel and
interval features. This judge is a differently built model (a ResNet-style CNN
trained end to end), so checks it confirms are less likely to exploit quirks of
the main oracle. It is also differentiable, which the gradient-based
explanation baselines need.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class _Block(nn.Module):
    def __init__(self, cin: int, cout: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(cin, cout, 7, padding=3), nn.BatchNorm1d(cout), nn.ReLU(),
            nn.Conv1d(cout, cout, 5, padding=2), nn.BatchNorm1d(cout), nn.ReLU(),
            nn.Conv1d(cout, cout, 3, padding=1), nn.BatchNorm1d(cout))
        self.skip = nn.Conv1d(cin, cout, 1)

    def forward(self, x):
        return torch.relu(self.net(x) + self.skip(x))


class ResNet1D(nn.Module):
    def __init__(self, C: int, n_classes: int, width: int = 64):
        super().__init__()
        self.body = nn.Sequential(_Block(C, width), _Block(width, 2 * width), _Block(2 * width, 2 * width))
        self.head = nn.Linear(2 * width, n_classes)

    def forward(self, x):
        return self.head(self.body(x).mean(-1))


class CNNJudge:
    """sklearn-like wrapper: fit / predict_proba on standardised (n, C, T) arrays."""

    def __init__(self, epochs: int = 30, lr: float = 1e-3, batch: int = 64, seed: int = 0):
        self.epochs, self.lr, self.batch, self.seed = epochs, lr, batch, seed
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def fit(self, X: np.ndarray, y: np.ndarray) -> "CNNJudge":
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        self.classes_ = np.unique(y)
        self.model = ResNet1D(X.shape[1], len(self.classes_)).to(self.device)
        opt = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, self.epochs)
        Xt = torch.as_tensor(X, dtype=torch.float32)
        yt = torch.as_tensor(np.searchsorted(self.classes_, y))
        for _ in range(self.epochs):
            self.model.train()
            perm = rng.permutation(len(X))
            for s in range(0, len(X), self.batch):
                ib = torch.as_tensor(perm[s:s + self.batch])
                loss = nn.functional.cross_entropy(self.model(Xt[ib].to(self.device)), yt[ib].to(self.device))
                opt.zero_grad()
                loss.backward()
                opt.step()
            sched.step()
        self.model.eval()
        return self

    @torch.no_grad()
    def predict_proba(self, X: np.ndarray, batch: int = 512) -> np.ndarray:
        self.model.eval()
        out = []
        for s in range(0, len(X), batch):
            xb = torch.as_tensor(X[s:s + batch], dtype=torch.float32, device=self.device)
            out.append(torch.softmax(self.model(xb), -1).float().cpu().numpy())
        return np.concatenate(out)

    def __getstate__(self):
        st = self.__dict__.copy()
        st["model"] = self.model.cpu()
        return st

    def __setstate__(self, st):
        self.__dict__.update(st)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.model.to(self.device)
