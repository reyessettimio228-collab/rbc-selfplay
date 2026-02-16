from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import random
import torch
from torch.utils.data import Dataset


class ReplayBuffer:
    def __init__(self, max_size: int = 200_000):
        self.max_size = max_size
        self.X: List[np.ndarray] = []
        self.P: List[np.ndarray] = []
        self.Z: List[float] = []
    def add(self, X: List[np.ndarray], P: List[np.ndarray], Z: List[float]):
        assert len(X) == len(P) == len(Z)
        self.X.extend(X)
        self.P.extend(P)
        self.Z.extend(Z)
        if len(self.X) > self.max_size:
            extra = len(self.X) - self.max_size
            self.X = self.X[extra:]
            self.P = self.P[extra:]
            self.Z = self.Z[extra:]
    def __len__(self) -> int:
        return len(self.X)
class BufferDataset(torch.utils.data.Dataset):
    def __init__(self, buf: ReplayBuffer):
        self.buf = buf
    def __len__(self) -> int:
        return len(self.buf)
    def __getitem__(self, idx: int):
        x = torch.from_numpy(self.buf.X[idx]).float()
        p = torch.from_numpy(self.buf.P[idx]).float()
        z = torch.tensor(self.buf.Z[idx], dtype=torch.float32)
        return x, p, z
