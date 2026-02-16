import chess
import torch
import reconchess


from src.config import DEVICE, Config
from src.player import *
from src.encoder import *
from src.search import *
from src.belief import *
from src.determinize import *
from src.sense import *



def selfplay_one_game(
    model: nn.Module,
    cfg: FastBotConfig,
    seed: int = 0,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[float], Optional[Color], str]:
    play_local_game = try_discover_play_local_game()
    bot_w = RBCFastAZPlayer(model=model, cfg=cfg, seed=seed)
    bot_b = RBCFastAZPlayer(model=model, cfg=cfg, seed=seed + 1)
    bot_w.store_training = True
    bot_b.store_training = True
    winner_color, reason, history = play_local_game(bot_w, bot_b)
    X = bot_w.X + bot_b.X
    P = bot_w.P + bot_b.P
    Z = bot_w.Z + bot_b.Z
    if len(Z) != len(X):
        if winner_color is None:
            z_w = 0.0
        else:
            z_w = 1.0 if bool(winner_color) == chess.WHITE else -1.0
        z_b = -z_w
        Z = [z_w] * len(bot_w.X) + [z_b] * len(bot_b.X)
    return X, P, Z, winner_color, reason


