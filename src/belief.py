import torch
import chess

PIECE_TYPES_6 = ["P", "N", "B", "R", "Q", "K"]
EMPTY = "EMPTY"
CHANNELS_7 = PIECE_TYPES_6 + [EMPTY]
C_BELIEF = 7
CH2I = {ch: i for i, ch in enumerate(CHANNELS_7)}
START_COUNTS = {"P": 8, "N": 2, "B": 2, "R": 2, "Q": 1, "K": 1}

def normalize_over_channels(B: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    s = B.sum(dim=0, keepdim=True).clamp_min(eps)
    return B / s

def piece_symbol_upper(pc: chess.Piece) -> str:
    return pc.symbol().upper()

def init_belief_from_initial(my_color: chess.Color) -> torch.Tensor:
    """
    RBC starts from the standard chess initial position.
    We initialize the opponent belief with a strong prior: probability 1 on the opponent’s
    initial piece placement, and EMPTY elsewhere.
    This is a reasonable baseline because the starting configuration is known in RBC;
    only after the game starts the opponent becomes hidden.
    """
    B = torch.zeros((C_BELIEF, 8, 8), dtype=torch.float32)
    B[CH2I[EMPTY]].fill_(1.0)
    board = chess.Board()
    opp_color = not my_color
    for sq, pc in board.piece_map().items():
        if pc.color != opp_color:
            continue
        r = chess.square_rank(sq)
        f = chess.square_file(sq)
        sym = piece_symbol_upper(pc)
        B[:, r, f] = 0.0
        B[CH2I[sym], r, f] = 1.0
    return normalize_over_channels(B)

def apply_sense_to_belief(
    B: torch.Tensor,
    sense_result: List[Tuple[Square, Optional[chess.Piece]]],
    my_color: chess.Color
) -> torch.Tensor:
    """Update opponent belief using ReconChess sense results (copy-based, not in-place)."""
    B2 = B.clone()
    for sq, pc in sense_result:
        r = chess.square_rank(sq)
        f = chess.square_file(sq)
        B2[:, r, f] = 0.0
        if pc is None or pc.color == my_color:
            B2[CH2I[EMPTY], r, f] = 1.0
        else:
            sym = piece_symbol_upper(pc)
            B2[CH2I[sym], r, f] = 1.0
    return normalize_over_channels(B2)
