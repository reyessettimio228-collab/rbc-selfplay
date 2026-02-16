import chess
import torch
import reconchess

from src.config import DEVICE
from src.utils import set_seeds
from src.encoder import *
from src.belief import *
from src.sense import *
from src.determinize import *
from src.search import *
from src.model import *
from src.encoding import *

@dataclass
class FastBotConfig:
    sims: int = 80
    c_puct: float = 1.5
class RBCFastAZPlayer(Player):
    def __init__(self, model: nn.Module, cfg: FastBotConfig, seed: int = 0):
        self.model = model
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)
        self.color: Optional[chess.Color] = None
        self.own_board: Optional[chess.Board] = None
        self.B: Optional[torch.Tensor] = None
        self.opp_counts: Optional[Dict[str,int]] = None
        self.store_training = False
        self.X: List[np.ndarray] = []
        self.P: List[np.ndarray] = []
        self.Z: List[float] = []
    def handle_game_start(self, color: Color, board: chess.Board, opponent_name: str):
        self.color = bool(color)
        self.own_board = chess.Board(None)
        self.own_board.clear()
        for sq, pc in board.piece_map().items():
            if pc.color == self.color:
                self.own_board.set_piece_at(sq, pc)
        self.own_board.turn = board.turn
        self.own_board.castling_rights = board.castling_rights
        self.B = init_belief_from_initial(self.color)
        self.opp_counts = dict(START_COUNTS)
    def handle_opponent_move_result(self, captured_my_piece: bool, capture_square: Optional[Square]):
        if captured_my_piece and capture_square is not None:
            if self.own_board is not None:
                self.own_board.remove_piece_at(capture_square)
    def choose_sense(self, sense_actions: List[Square], move_actions: List[chess.Move], seconds_left: float) -> Square:
        assert self.B is not None
        return choose_sense_square_entropy(self.B, sense_actions)
    def handle_sense_result(self, sense_result: List[Tuple[Square, Optional[chess.Piece]]]):
        assert self.B is not None and self.color is not None
        self.B = apply_sense_to_belief(self.B, sense_result, self.color)
    def choose_move(self, move_actions: List[chess.Move], seconds_left: float) -> Optional[chess.Move]:
        assert self.own_board is not None and self.color is not None and self.B is not None and self.opp_counts is not None
        if not move_actions:
            return None
        x_root = encode_state(self.own_board, self.color, self.B)
        det = greedy_determinize_opponent(self.B, self.own_board, self.color, self.opp_counts)
        det.turn = self.own_board.turn
        N_a = puct_root(
            model=self.model,
            x_root=x_root,
            det_board=det,
            move_actions=move_actions,
            sims=self.cfg.sims,
            c_puct=self.cfg.c_puct,
            device=DEVICE,
        )
        if self.store_training:
            self.X.append(x_root.detach().cpu().numpy().astype(np.float32))
            self.P.append(visits_to_policy_target(N_a))
        if N_a:
            best_a = max(N_a.items(), key=lambda kv: kv[1])[0]
            mv = index_to_move(best_a)
            if mv in move_actions:
                return mv
        return move_actions[int(self.rng.integers(0, len(move_actions)))]
    def handle_move_result(
        self,
        requested_move: Optional[chess.Move],
        taken_move: Optional[chess.Move],
        captured_opponent_piece: bool,
        capture_square: Optional[Square],
    ):
        if self.own_board is None or self.color is None:
            return
        if taken_move is not None:
            apply_taken_move_to_own_board(self.own_board, taken_move, self.color)
        if captured_opponent_piece and capture_square is not None and self.opp_counts is not None and self.B is not None:
            r = chess.square_rank(capture_square)
            f = chess.square_file(capture_square)
            probs = self.B[:, r, f].detach().cpu()
            best_sym = None
            best_p = -1.0
            for sym in PIECE_TYPES_6:
                p = float(probs[CH2I[sym]].item())
                if p > best_p and self.opp_counts.get(sym, 0) > 0:
                    best_p = p
                    best_sym = sym
            if best_sym is None:
                if self.opp_counts.get("P", 0) > 0:
                    best_sym = "P"
                else:
                    for sym in ["N", "B", "R", "Q", "K"]:
                        if self.opp_counts.get(sym, 0) > 0:
                            best_sym = sym
                            break
            if best_sym is not None:
                self.opp_counts[best_sym] = max(0, self.opp_counts.get(best_sym, 0) - 1)
    def handle_game_end(self, winner_color: Optional[Color], reason: str, game_history: Any):
        if not self.store_training or self.color is None:
            return
        if winner_color is None:
            z = 0.0
        else:
            z = 1.0 if bool(winner_color) == self.color else -1.0
        self.Z = [z] * len(self.X)
