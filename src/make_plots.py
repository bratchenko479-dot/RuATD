import os
import pandas as pd
import matplotlib.pyplot as plt

os.makedirs("results/figures", exist_ok=True)

train = pd.read_csv("data/processed/train.csv")
results = pd.read_csv("results/results.csv")

# 1. Class distribution
train["label"].value_counts().sort_index().plot(kind="bar")
plt.title("Class distribution in train set")
plt.xlabel("Class")
plt.ylabel("Number of texts")
plt.xticks([0, 1], ["Human", "Machine"], rotation=0)
plt.tight_layout()
plt.savefig("results/figures/class_distribution.png", dpi=300)
plt.close()

# 2. Text length distribution
train["length"] = train["text"].str.split().str.len()
train["length"].hist(bins=50)
plt.title("Text length distribution")
plt.xlabel("Number of words")
plt.ylabel("Number of texts")
plt.tight_layout()
plt.savefig("results/figures/text_length_distribution.png", dpi=300)
plt.close()

# 3. Model comparison
results.plot(
    x="model",
    y="macro_f1",
    kind="bar",
    legend=False
)
plt.title("Model comparison by Macro-F1")
plt.xlabel("Model")
plt.ylabel("Macro-F1")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig("results/figures/model_comparison.png", dpi=300)
plt.close()

print("Saved figures to results/figures/")