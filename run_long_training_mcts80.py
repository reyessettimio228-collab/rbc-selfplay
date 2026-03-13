from rbc_selfplay import run_selfplay_training
import time

out_dir = f"runs/long_mcts80_{time.strftime('%Y%m%d_%H%M%S')}"

run_selfplay_training(
    iters=900,
    games_per_iter=4,
    train_steps_per_iter=200,
    batch_size=64,
    sims_selfplay=80,
    sims_eval=40,
    eval_games=30,
    save_every=150,
    keep_last=3,
    seed=0,
    out_dir=out_dir,
    results_csv="results.csv",
)

print("OUT_DIR:", out_dir)
