import pandas as pd
import matplotlib.pyplot as plt
def plot_results(csv_path: str):
    df = pd.read_csv(csv_path)
    if len(df) == 0:
        print("Empty CSV.")
        return df
    plt.figure()
    plt.plot(df["iter"], df["loss"])
    plt.xlabel("iter")
    plt.ylabel("loss")
    plt.title("Training loss")
    plt.show()
    plt.figure()
    total = (df["eval_wins"] + df["eval_losses"] + df["eval_draws"]).clip(lower=1)
    winrate = df["eval_wins"] / total
    plt.plot(df["iter"], winrate)
    plt.xlabel("iter")
    plt.ylabel("winrate vs random (eval)")
    plt.title("Winrate vs RandomBot")
    plt.show()
    return df
'''Example:
df = plot_results("fast_checkpoints/results.csv")'''
