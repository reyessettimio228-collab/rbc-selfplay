from src.config import Config
from src.selfplay import generate_selfplay_data
from src.train_loop import train_from_data
from src.eval import evaluate


def main() -> None:
    cfg = Config()

    data = generate_selfplay_data(cfg)
    model = train_from_data(cfg, data)
    metrics = evaluate(cfg, model)

    print("Done. Metrics:", metrics)


if __name__ == "__main__":
    main()
