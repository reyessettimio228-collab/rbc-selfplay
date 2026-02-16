import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import DEVICE
from src.encoding import POLICY_SIZE

class FastPolicyValueNet(nn.Module):
    def __init__(self, in_ch: int = 15, trunk: int = 64):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Conv2d(in_ch, trunk, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(trunk, trunk, 3, padding=1),
            nn.ReLU(),
        )
        self.pol = nn.Sequential(nn.Conv2d(trunk, 32, 1), nn.ReLU())
        self.pol_fc = nn.Linear(32*8*8, POLICY_SIZE)
        self.val = nn.Sequential(nn.Conv2d(trunk, 16, 1), nn.ReLU())
        self.val_fc1 = nn.Linear(16*8*8, 64)
        self.val_fc2 = nn.Linear(64, 1)
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.trunk(x)
        logits = self.pol_fc(self.pol(h).flatten(1))
        v = torch.tanh(self.val_fc2(F.relu(self.val_fc1(self.val(h).flatten(1))))).squeeze(-1)
        return logits, v
