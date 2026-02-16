import torch
import chess
from src.config import DEVICE

PIECE_ORDER = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]
def board_to_own_planes(board: chess.Board, my_color: chess.Color) -> torch.Tensor:
    X = torch.zeros((6, 8, 8), dtype=torch.float32)
    for sq, pc in board.piece_map().items():
        if pc.color != my_color:
            continue
        r = chess.square_rank(sq)
        f = chess.square_file(sq)
        i = PIECE_ORDER.index(pc.piece_type)
        X[i, r, f] = 1.0
    return X
def metadata_planes(board: chess.Board, my_color: chess.Color) -> torch.Tensor:
    turn_plane = torch.full((1,8,8), 1.0 if board.turn == my_color else 0.0, dtype=torch.float32)
    ply = min(board.fullmove_number * 2, 200) / 200.0
    ply_plane = torch.full((1,8,8), float(ply), dtype=torch.float32)
    return torch.cat([turn_plane, ply_plane], dim=0)
def encode_state(own_board: chess.Board, my_color: chess.Color, B: torch.Tensor) -> torch.Tensor:
    own = board_to_own_planes(own_board, my_color)
    meta = metadata_planes(own_board, my_color)
    return torch.cat([own, B, meta], dim=0)
