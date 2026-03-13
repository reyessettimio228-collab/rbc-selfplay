import csv
import os
import torch
import chess

from reconchess.bots.trout_bot import TroutBot
from reconchess.bots.attacker_bot import AttackerBot

from rbc_selfplay import (
    FastPolicyValueNet,
    RBCFastAZPlayer,
    FastBotConfig,
    load_checkpoint,
    try_discover_play_local_game
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
play_local_game = try_discover_play_local_game()


def make_agent(ckpt, sims):

    model = FastPolicyValueNet(in_ch=15, trunk=64).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters())

    load_checkpoint(ckpt, model, opt)
    model.eval()

    return RBCFastAZPlayer(
        model=model,
        cfg=FastBotConfig(sims=sims, c_puct=1.5),
        seed=0
    )


def run_match(agent_name, agent, opponent_name, opponent, games, outcsv):

    wins = losses = draws = 0

    for g in range(games):

        white = agent if g % 2 == 0 else opponent
        black = opponent if g % 2 == 0 else agent

        winner, reason, _ = play_local_game(white, black)

        if winner is None:
            draws += 1
        elif bool(winner) == (g % 2 == 0):
            wins += 1
        else:
            losses += 1

        print(agent_name, "vs", opponent_name, g+1, "/", games)

    with open(outcsv, "a") as f:
        writer = csv.writer(f)
        writer.writerow([agent_name, opponent_name, wins, losses, draws])


def main():

    ckpt1 = "runs/long_20260302_192254/ckpt_iter_2500.pt"
    ckpt2 = "runs/long_mcts80_20260306_235549/latest.pt"

    agentA = make_agent(ckpt1, sims=80)
    agentB = make_agent(ckpt2, sims=80)

    opponents = [
        ("AttackerBot", AttackerBot())
    ]

    os.makedirs("runs/final_eval", exist_ok=True)

    with open("runs/final_eval/results.csv","w") as f:
        writer = csv.writer(f)
        writer.writerow(["agent","opponent","wins","losses","draws"])

    for name,bot in opponents:

        run_match("AZ40", agentA, name, bot, 50, "runs/final_eval/results.csv")
        run_match("AZ80", agentB, name, bot, 50, "runs/final_eval/results.csv")

    run_match("AZ40", agentA, "AZ80", agentB, 50, "runs/final_eval/results.csv")
if __name__ == "__main__":
    main()
