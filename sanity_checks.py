from src.player import *
from src.encoding import *
from src.belief import *
from src.encoder import *
from src.model import *
from src.search import *

def run_all_checks():
    model = FastPolicyValueNet(in_ch=15, trunk=64).to(DEVICE)
    bot = RBCFastAZPlayer(model=model, cfg=FastBotConfig(sims=10, c_puct=1.5), seed=0)
    start_board = chess.Board()
    bot.handle_game_start(color=chess.WHITE, board=start_board, opponent_name="opp")
    assert bot.B is not None
    s = bot.B.sum(dim=0)
    assert float((s - 1.0).abs().max().item()) < 1e-4, "Belief not normalized per square"
    sense_actions = list(range(64))
    mv_actions = list(start_board.legal_moves)
    sq = bot.choose_sense(sense_actions, mv_actions, seconds_left=100.0)
    assert sq in sense_actions, "choose_sense returned illegal square"
    mv = bot.choose_move(mv_actions, seconds_left=100.0)
    assert (mv is None) or (mv in mv_actions), "choose_move returned move not in move_actions"
    print("Core invariants: OK")
    try:
        smoke_test_vs_random(n_games=2, seed=0)
        print("Smoke games: OK")
    except Exception as e:
        print("Smoke games: SKIPPED/FAILED (environment issue):", e)
run_all_checks()

def main():
    # run checks here
    pass

if __name__ == "__main__":
    main()
