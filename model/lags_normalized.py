import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data/processed/tesla_combined.csv")
df["date"] = pd.to_datetime(df["date"])

df["p_sum"] = df["avg_sentiment"] * df["tweet_count"]

for lag in range(1, 8):
    df[f"p_sum_lag_{lag}"] = df["p_sum"].shift(lag)

def normalize(series):
    return (series - series.min()) / (series.max() - series.min())

df["Adj_Close_norm"] = normalize(df["Adj Close"])
for i in range(1, 8):
    df[f"p_sum_lag_{i}_norm"] = normalize(df[f"p_sum_lag_{i}"])

df.to_csv("data/processed/tesla_lagged.csv", index=False)

plt.figure(figsize=(18, 10))
plt.plot(df["date"], df["Adj_Close_norm"], label="Adj Close (norm)", color="black", linewidth=2)
colors = ['blue', 'green', 'red', 'orange', 'purple', 'brown', 'cyan']
for i in range(1, 8):
    plt.plot(df["date"], df[f"p_sum_lag_{i}_norm"], label=f"p_sum_lag_{i} (norm)", linestyle='--', color=colors[i-1])
plt.title("Adj Close (norm) vs. p_sum_lag_1–7 (norm)", fontsize=16)
plt.xlabel("Data")
plt.ylabel("Valoare normalizată")
plt.xticks(rotation=45)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

for i in range(1, 8):
    plt.figure(figsize=(16, 5))
    plt.plot(df["date"], df["Adj_Close_norm"], label="Adj Close (norm)", color="black", linewidth=2)
    plt.plot(df["date"], df[f"p_sum_lag_{i}_norm"], label=f"p_sum_lag_{i} (norm)", linestyle='--', color="orange")
    plt.title(f"Adj Close (norm) vs. p_sum_lag_{i} (norm)", fontsize=15)
    plt.xlabel("Data")
    plt.ylabel("Valoare normalizată")
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
