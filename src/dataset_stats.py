import os
import pandas as pd

os.makedirs("results", exist_ok=True)

train = pd.read_csv("data/processed/train.csv")
test = pd.read_csv("data/processed/test.csv")

train["split"] = "train"
test["split"] = "evaluation"

df = pd.concat([train, test], ignore_index=True)

# =========================
# 1. Dataset size by split
# =========================

dataset_rows = []

for split_name, split_df in df.groupby("split"):
    human_count = (split_df["label"] == 0).sum()
    machine_count = (split_df["label"] == 1).sum()
    total_count = len(split_df)

    dataset_rows.append({
        "split": split_name,
        "human": human_count,
        "machine": machine_count,
        "total": total_count,
        "human_percent": human_count / total_count,
        "machine_percent": machine_count / total_count
    })

dataset_stats = pd.DataFrame(dataset_rows)
dataset_stats.to_csv("results/dataset_stats.csv", index=False)

# =========================
# 2. Text length statistics
# =========================

df["word_length"] = df["text"].astype(str).str.split().str.len()
df["char_length"] = df["text"].astype(str).str.len()

length_rows = []

for label_value, label_name in [(0, "human"), (1, "machine")]:
    label_df = df[df["label"] == label_value]

    length_rows.append({
        "class": label_name,
        "avg_word_length": label_df["word_length"].mean(),
        "median_word_length": label_df["word_length"].median(),
        "avg_char_length": label_df["char_length"].mean(),
        "median_char_length": label_df["char_length"].median()
    })

text_length_stats = pd.DataFrame(length_rows)
text_length_stats.to_csv("results/text_length_stats.csv", index=False)

# =========================
# 3. Print results
# =========================

print("Dataset statistics:")
print(dataset_stats)

print("\nText length statistics:")
print(text_length_stats)

print("\nSaved:")
print("results/dataset_stats.csv")
print("results/text_length_stats.csv")