import pandas as pd

df = pd.read_csv("runs/final_eval/results.csv")

ratings = {
    "TroutBot": 1100.0,
    "AttackerBot": 1050.0,
    "AZ40": 1000.0,
    "AZ80": 1000.0,
}

for _, r in df.iterrows():
    a = r.agent
    b = r.opponent
    score = (r.wins + 0.5 * r.draws) / (r.wins + r.losses + r.draws)

    ra = ratings[a]
    rb = ratings[b]

    expected = 1 / (1 + 10 ** ((rb - ra) / 400))
    ra = ra + 32 * (score - expected)
    ratings[a] = ra

print("\nFINAL ELO\n")
for k, v in sorted(ratings.items(), key=lambda x: -x[1]):
    print(k, round(v, 1))

