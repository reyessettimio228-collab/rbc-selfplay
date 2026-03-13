from rbc_selfplay import run_selfplay_training
import time

out_dir = f"runs/long_{time.strftime('%Y%m%d_%H%M%S')}"

run_selfplay_training(
    iters=2600,
    games_per_iter=4,
    train_steps_per_iter=200,
    batch_size=64,
    sims_selfplay=40,
    sims_eval=25,
    eval_games=30,
    save_every=250,
    keep_last=3,
    seed=0,
    out_dir=out_dir,
    results_csv="results.csv",
)

print("OUT_DIR:", out_dir)
