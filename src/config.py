from dataclasses import dataclass


@dataclass
class Config:
    seed: int = 0

    # Self-play
    num_selfplay_games: int = 10
    max_moves_per_game: int = 200

    # Training
    batch_size: int = 64
    learning_rate: float = 1e-4
    num_epochs: int = 1

    # Paths (keep lightweight; heavy artifacts are gitignored)
    data_dir: str = "data"
    models_dir: str = "models"
    results_dir: str = "results"

