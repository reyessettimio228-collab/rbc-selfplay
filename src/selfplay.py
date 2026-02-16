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


def run_selfplay_training(
    iters: int = 5,
    games_per_iter: int = 6,
    train_steps_per_iter: int = 300,
    batch_size: int = 64,
    sims_selfplay: int = 40,
    sims_eval: int = 20,
    seed: int = 0,
    out_dir: str = "fast_checkpoints",
    results_csv: str = "results.csv",
):
    os.makedirs(out_dir, exist_ok=True)
    results_path = os.path.join(out_dir, results_csv)
    fieldnames = [
        "timestamp", "iter", "buffer_size",
        "loss", "loss_policy", "loss_value",
        "eval_wins", "eval_losses", "eval_draws",
        "sims_selfplay", "sims_eval",
        "games_per_iter", "train_steps_per_iter", "batch_size",
        "ckpt_path",
    ]
    if not os.path.exists(results_path):
        with open(results_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
    model = FastPolicyValueNet(in_ch=15, trunk=64).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    buf = ReplayBuffer(max_size=200_000)
    step = 0
    for it in range(iters):
        cfg = FastBotConfig(sims=sims_selfplay, c_puct=1.5)
        for g in range(games_per_iter):
            X, P, Z, winner, reason = selfplay_one_game(model, cfg, seed=seed + it * 1000 + g)
            buf.add(X, P, Z)
        ds = BufferDataset(buf)
        dl = torch.utils.data.DataLoader(
            ds, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True
        )
        metrics = train_steps(model, opt, dl, device=DEVICE, steps=train_steps_per_iter)
        try:
            wins = 0
            losses = 0
            draws = 0
            play_local_game = try_discover_play_local_game()
            for eg in range(6):
                bot = RBCFastAZPlayer(
                    model=model,
                    cfg=FastBotConfig(sims=sims_eval, c_puct=1.5),
                    seed=seed + 999 + it * 10 + eg,
                )
                opp = reconchess.bots.random_bot.RandomBot()
                winner_color, reason, history = play_local_game(bot, opp)
                if winner_color is None:
                    draws += 1
                elif bool(winner_color) == chess.WHITE:
                    wins += 1
                else:
                    losses += 1
            eval_res = {"wins": wins, "losses": losses, "draws": draws}
        except Exception as e:
            eval_res = {"wins": 0, "losses": 0, "draws": 0}
            print("Eval skipped/failed:", e)
        ckpt_path = os.path.join(out_dir, f"ckpt_iter_{it}.pt")
        save_checkpoint(
            ckpt_path,
            model,
            opt,
            step,
            extra={"iter": it, "buffer": len(buf)},
        )
        print(f"[iter {it}] buffer={len(buf)} train={metrics} eval6={eval_res} saved={ckpt_path}")
        row = {
            "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
            "iter": it,
            "buffer_size": len(buf),
            "loss": metrics.get("loss"),
            "loss_policy": metrics.get("lp"),
            "loss_value": metrics.get("lv"),
            "eval_wins": eval_res.get("wins", 0),
            "eval_losses": eval_res.get("losses", 0),
            "eval_draws": eval_res.get("draws", 0),
            "sims_selfplay": sims_selfplay,
            "sims_eval": sims_eval,
            "games_per_iter": games_per_iter,
            "train_steps_per_iter": train_steps_per_iter,
            "batch_size": batch_size,
            "ckpt_path": ckpt_path,
        }
        with open(results_path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writerow(row)
        step += 1
    return model


