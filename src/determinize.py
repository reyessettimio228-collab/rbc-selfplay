import chess
import random

def greedy_determinize_opponent(
    B: torch.Tensor,
    own_board: chess.Board,
    my_color: chess.Color,
    opp_counts: Dict[str, int],
) -> chess.Board:
    det = chess.Board(None)
    det.clear()
    for sq, pc in own_board.piece_map().items():
        det.set_piece_at(sq, pc)
    occupied = set(own_board.piece_map().keys())
    opp_color = not my_color
    for sym in PIECE_TYPES_6:
        k = int(opp_counts.get(sym, 0))
        if k <= 0:
            continue
        probs = []
        ch = CH2I[sym]
        for sq in chess.SQUARES:
            if sq in occupied:
                continue
            r = chess.square_rank(sq)
            f = chess.square_file(sq)
            probs.append((float(B[ch, r, f].item()), sq))
        probs.sort(reverse=True, key=lambda x: x[0])
        placed = 0
        for _, sq in probs:
            if sq in occupied:
                continue
            psym = sym.lower() if opp_color == chess.BLACK else sym
            det.set_piece_at(sq, chess.Piece.from_symbol(psym))
            occupied.add(sq)
            placed += 1
            if placed >= k:
                break
    det.turn = own_board.turn
    det.castling_rights = own_board.castling_rights
    return det

def apply_taken_move_to_own_board(own_board: chess.Board, mv: chess.Move, my_color: chess.Color) -> None:
    pc = own_board.piece_at(mv.from_square)
    if pc is None or pc.color != my_color:
        return
    own_board.remove_piece_at(mv.from_square)
    if mv.promotion is not None and pc.piece_type == chess.PAWN:
        pc = chess.Piece(mv.promotion, my_color)
    own_board.set_piece_at(mv.to_square, pc)
    if pc.piece_type == chess.KING:
        f_from = chess.square_file(mv.from_square)
        f_to = chess.square_file(mv.to_square)
        r_rank = chess.square_rank(mv.from_square)
        if abs(f_to - f_from) == 2:
            if f_to > f_from:
                rook_from = chess.square(7, r_rank)
                rook_to = chess.square(5, r_rank)
            else:
                rook_from = chess.square(0, r_rank)
                rook_to = chess.square(3, r_rank)
            rook = own_board.piece_at(rook_from)
            if rook is not None and rook.color == my_color and rook.piece_type == chess.ROOK:
                own_board.remove_piece_at(rook_from)
                own_board.set_piece_at(rook_to, rook)
    own_board.turn = not own_board.turn
    if my_color == chess.BLACK:
        own_board.fullmove_number += 1
