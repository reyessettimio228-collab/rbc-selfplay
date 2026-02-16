from src.player import *

def try_discover_play_local_game():
    try:
        from reconchess.utilities import play_local_game
        return play_local_game
    except Exception:
        pass
    try:
        from reconchess.scripts.rc_play_game import play_local_game
        return play_local_game
    except Exception:
        pass
    raise ImportError(
        "Could not find play_local_game in reconchess. Install a version that provides local runner utilities."
    )
def smoke_test_vs_random(n_games: int = 5, seed: int = 0):
    play_local_game = try_discover_play_local_game()
    model = FastPolicyValueNet(in_ch=15, trunk=64).to(DEVICE)
    cfg = FastBotConfig(sims=40, c_puct=1.5)
    wins = 0
    losses = 0
    draws = 0
    for g in range(n_games):
        bot = RBCFastAZPlayer(model=model, cfg=cfg, seed=seed + g)
        opp = reconchess.bots.random_bot.RandomBot()
        winner_color, reason, history = play_local_game(bot, opp)
        if winner_color is None:
            draws += 1
        elif bool(winner_color) == chess.WHITE:
            wins += 1
        else:
            losses += 1
        print(f"[game {g}] winner={winner_color} reason={reason}")
    print({"wins": wins, "losses": losses, "draws": draws})

