import os
from typing import Any, Dict, Optional

import torch

def save_checkpoint(
    path: str,
    model: nn.Module,
    opt: torch.optim.Optimizer,
    step: int,
    extra: Optional[dict] = None
) -> None:
    ckpt = {
        "model": model.state_dict(),
        "opt": opt.state_dict(),
        "step": int(step),
        "extra": extra or {},
    }
    torch.save(ckpt, path)
def load_checkpoint(
    path: str,
    model: nn.Module,
    opt: Optional[torch.optim.Optimizer] = None
) -> dict:
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    if opt is not None and "opt" in ckpt:
        opt.load_state_dict(ckpt["opt"])
    return ckpt

def save_checkpoint(path: str, model: nn.Module, opt: torch.optim.Optimizer, step: int, extra: Optional[dict] = None) -> None:
    ckpt = {"model": model.state_dict(), "opt": opt.state_dict(), "step": int(step), "extra": extra or {}}
    torch.save(ckpt, path)
def load_checkpoint(path: str, model: nn.Module, opt: Optional[torch.optim.Optimizer] = None) -> dict:
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    if opt is not None and "opt" in ckpt:
        opt.load_state_dict(ckpt["opt"])
    return ckpt
