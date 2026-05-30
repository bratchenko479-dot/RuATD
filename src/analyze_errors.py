import os
import pandas as pd

errors_path = "results/errors_best_model.csv"
fp_path = "results/false_positives.csv"
fn_path = "results/false_negatives.csv"

if not os.path.exists(errors_path):
    raise FileNotFoundError(
        "results/errors_best_model.csv not found. "
        "Run python src/train_baselines.py first."
    )

errors = pd.read_csv(errors_path)
false_positives = pd.read_csv(fp_path)
false_negatives = pd.read_csv(fn_path)

print("Error analysis for the best model")
print("=" * 80)

print("Total errors:", len(errors))
print("False positives:", len(false_positives))
print("False negatives:", len(false_negatives))

print("\nFalse positives:")
print("Gold label = 0 human, prediction = 1 machine")
print("=" * 80)

for i, row in false_positives.head(5).iterrows():
    print("TEXT:")
    print(str(row["text"])[:700])
    print("-" * 80)

print("\nFalse negatives:")
print("Gold label = 1 machine, prediction = 0 human")
print("=" * 80)

for i, row in false_negatives.head(5).iterrows():
    print("TEXT:")
    print(str(row["text"])[:700])
    print("-" * 80)

summary = pd.DataFrame([{
    "total_errors": len(errors),
    "false_positives": len(false_positives),
    "false_negatives": len(false_negatives)
}])

summary.to_csv("results/error_summary.csv", index=False)

print("\nSaved:")
print("results/error_summary.csv")