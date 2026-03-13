import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("runs/final_eval/results.csv")
df["score"] = (df.wins + 0.5 * df.draws) / (df.wins + df.losses + df.draws)

plt.figure(figsize=(8, 5))
labels = [f"{r.agent} vs {r.opponent}" for _, r in df.iterrows()]
scores = list(df["score"])

plt.bar(labels, scores)
plt.ylabel("Score")
plt.title("Agent performance against RBC baseline opponents")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.savefig("runs/final_eval/evaluation_plot.png", dpi=300)
plt.savefig("runs/final_eval/evaluation_plot.pdf")
print("saved plots")
