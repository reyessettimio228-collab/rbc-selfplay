from typing import Dict, Any, List
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

def train_steps(
    model: nn.Module,
    opt: torch.optim.Optimizer,
    dl: torch.utils.data.DataLoader,
    device: str,
    steps: int = 200,
) -> Dict[str, float]:
    model.train()
    it = iter(dl)
    ema = {"loss": None, "lp": None, "lv": None}
    for _ in range(steps):
        try:
            x, p, z = next(it)
        except StopIteration:
            it = iter(dl)
            x, p, z = next(it)
        x = x.to(device)
        p = p.to(device)
        z = z.to(device)
        opt.zero_grad(set_to_none=True)
        logits, v_pred = model(x)
        logp = F.log_softmax(logits, dim=-1)
        loss_policy = F.kl_div(logp, p, reduction="batchmean")
        loss_value = F.mse_loss(v_pred, z)
        loss = loss_policy + loss_value
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        for k, val in [("loss", loss.item()), ("lp", loss_policy.item()), ("lv", loss_value.item())]:
            if ema[k] is None:
                ema[k] = float(val)
            else:
                ema[k] = 0.98 * ema[k] + 0.02 * float(val)
    return {k: float(v) for k, v in ema.items()}


def train_from_data(cfg: Config, data: List[Dict[str, Any]]) -> Any:
    set_seeds(cfg.seed)
    raise NotImplementedError("train_from_data not wired yet. Use train_steps(model,opt,dl,DEVICE).")


