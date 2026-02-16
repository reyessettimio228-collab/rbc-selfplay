from typing import Any, Dict, List

from src.config import Config
from src.utils import set_seed


def train_from_data(cfg: Config, data: List[Dict[str, Any]]) -> Any:
    """
    Train your policy/value network from self-play data.
    Return the trained model (or a dict of weights).
    """
    set_seed(cfg.seed)

    model: Any = None
    # TODO: move training loop from notebook here
    return model

