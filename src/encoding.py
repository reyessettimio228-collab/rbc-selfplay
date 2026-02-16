PROMO_TO_ID = {None: 0, chess.KNIGHT: 1, chess.BISHOP: 2, chess.ROOK: 3, chess.QUEEN: 4}
ID_TO_PROMO = {v: k for k, v in PROMO_TO_ID.items()}
POLICY_SIZE = 64 * 64 * 5
def move_to_index(mv: chess.Move) -> int:
    f = mv.from_square
    t = mv.to_square
    pid = PROMO_TO_ID.get(mv.promotion, 0)
    return (f * 64 + t) * 5 + pid
def index_to_move(idx: int) -> chess.Move:
    pid = idx % 5
    x = idx // 5
    f = x // 64
    t = x % 64
    promo = ID_TO_PROMO.get(pid, None)
    return chess.Move(from_square=f, to_square=t, promotion=promo)
