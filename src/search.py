import math
import torch
from src.config import DEVICE
from src.encoding import move_to_index

from src.model import FastPolicyValueNet

@torch.no_grad()
def nn_priors_and_value(model: nn.Module, x: torch.Tensor, legal_moves: List[chess.Move], device: str) -> Tuple[Dict[int,float], float]:
    model.eval()
    xb = x.unsqueeze(0).to(device)
    logits, v = model(xb)
    logits = logits[0].cpu()
    v = float(v.item())
    idxs = [move_to_index(m) for m in legal_moves]
    l = logits[idxs]
    p = torch.softmax(l, dim=0).numpy()
    pri = {idxs[i]: float(p[i]) for i in range(len(idxs))}
    return pri, v
@torch.no_grad()
def puct_root(
    model: nn.Module,
    x_root: torch.Tensor,
    det_board: chess.Board,
    move_actions: List[chess.Move],
    sims: int = 80,
    c_puct: float = 1.5,
    device: str = "cpu",
) -> Dict[int,int]:
    if not move_actions:
        return {}
    priors, _ = nn_priors_and_value(model, x_root, move_actions, device=device)
    N = 0
    N_a: Dict[int,int] = {move_to_index(m): 0 for m in move_actions}
    W_a: Dict[int,float] = {move_to_index(m): 0.0 for m in move_actions}
    def Q(a):
        n = N_a[a]
        return 0.0 if n == 0 else W_a[a] / n
    for _ in range(sims):
        N += 1
        best_a = None
        best_s = -1e18
        for m in move_actions:
            a = move_to_index(m)
            u = c_puct * priors.get(a, 0.0) * math.sqrt(N) / (1 + N_a[a])
            s = Q(a) + u
            if s > best_s:
                best_s = s
                best_a = a
        mv = index_to_move(best_a)
        b2 = det_board.copy()
        if mv not in b2.legal_moves:
            v_leaf = -1.0
        else:
            b2.push(mv)
            own2 = chess.Board(None); own2.clear()
            for sq, pc in b2.piece_map().items():
                if pc.color == det_board.turn:
                    pass
            _, v = model(x_root.unsqueeze(0).to(device))
            v_leaf = float(v.item())
        N_a[best_a] += 1
        W_a[best_a] += float(v_leaf)
    return N_a
def visits_to_policy_target(N_a: Dict[int,int]) -> np.ndarray:
    P = np.zeros((POLICY_SIZE,), dtype=np.float32)
    if not N_a:
        return P
    total = sum(N_a.values())
    for a, n in N_a.items():
        P[a] = n / max(1, total)
    return P
