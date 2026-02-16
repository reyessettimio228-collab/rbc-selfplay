from typing import Any, Dict, List

from src.config import Config
from src.utils import set_seed


def generate_selfplay_data(cfg: Config) -> List[Dict[str, Any]]:
    """
    Return a list of training samples produced by self-play.
    Each sample can be a dict like:
      {"obs": ..., "policy_target": ..., "value_target": ...}
    """
    set_seed(cfg.seed)

    data: List[Dict[str, Any]] = []
    # TODO: move your self-play logic from the notebook here
    # for _ in range(cfg.num_selfplay_games):
    #     ...
    return data

