import pandas as pd
import os

baseline_path = "results/results_baselines.csv"
rubert_path = "results/results_rubert.csv"

baselines = pd.read_csv(baseline_path)
rubert = pd.read_csv(rubert_path)

results = pd.concat([baselines, rubert], ignore_index=True)
results = results.sort_values("macro_f1", ascending=False)

results.to_csv("results/results.csv", index=False)

print(results)
print("\nSaved final results to results/results.csv")