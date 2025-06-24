import pandas as pd

df = pd.read_csv("data/processed/tesla_lagged.csv")

split_idx = int(len(df) * 0.96)

df_80 = df.iloc[:split_idx]
df_20 = df.iloc[split_idx:]

df_80.to_csv("data/processed/tesla_lagged_96.csv", index=False)
df_20.to_csv("data/processed/tesla_lagged_4.csv", index=False)