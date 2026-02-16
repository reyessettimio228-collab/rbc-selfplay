import chess
import torch  # se usi tensori / rete
from src.config import DEVICE  # solo se fai .to(DEVICE)

SENSE_OFFSETS = [(dr, df) for dr in (-1,0,1) for df in (-1,0,1)]
def square_entropy(p: torch.Tensor, eps: float = 1e-8) -> float:
    p = p.clamp_min(eps)
    return float(-(p * p.log()).sum().item())
def choose_sense_square_entropy(B: torch.Tensor, sense_actions: List[Square]) -> Square:
    """Choose among *allowed* sense_actions provided by ReconChess."""
    best_sq = sense_actions[0]
    best_e = -1e18
    for sq in sense_actions:
        r0 = chess.square_rank(sq)
        f0 = chess.square_file(sq)
        tot = 0.0
        for dr, df in SENSE_OFFSETS:
            r = r0 + dr; f = f0 + df
            if 0 <= r < 8 and 0 <= f < 8:
                tot += square_entropy(B[:, r, f])
        if tot > best_e:
            best_e = tot
            best_sq = sq
    return best_sq
