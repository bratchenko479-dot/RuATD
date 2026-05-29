import os
import pandas as pd

from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report

os.makedirs("results", exist_ok=True)

train = pd.read_csv("data/processed/train.csv")
test = pd.read_csv("data/processed/test.csv")

x_train = train["text"]
y_train = train["label"]

x_test = test["text"]
y_test = test["label"]


def evaluate_model(name, model):
    print("=" * 80)
    print(name)

    model.fit(x_train, y_train)
    predictions = model.predict(x_test)

    accuracy = accuracy_score(y_test, predictions)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test,
        predictions,
        average="macro",
        zero_division=0
    )

    print("Accuracy:", accuracy)
    print("Macro Precision:", precision)
    print("Macro Recall:", recall)
    print("Macro F1:", f1)
    print()
    print(classification_report(y_test, predictions))

    return {
        "model": name,
        "accuracy": accuracy,
        "macro_precision": precision,
        "macro_recall": recall,
        "macro_f1": f1
    }


results = []

majority_model = DummyClassifier(strategy="most_frequent")

results.append(evaluate_model("Majority baseline", majority_model))


logreg_model = Pipeline([
    ("tfidf", TfidfVectorizer(
        lowercase=True,
        max_features=50000,
        ngram_range=(1, 2)
    )),
    ("classifier", LogisticRegression(
        max_iter=1000,
        class_weight="balanced"
    ))
])

results.append(evaluate_model("TF-IDF + Logistic Regression", logreg_model))


svm_model = Pipeline([
    ("tfidf", TfidfVectorizer(
        lowercase=True,
        max_features=50000,
        ngram_range=(1, 2)
    )),
    ("classifier", LinearSVC(
        class_weight="balanced"
    ))
])

results.append(evaluate_model("TF-IDF + Linear SVM", svm_model))


results_df = pd.DataFrame(results)
results_df.to_csv("results/results_baselines.csv", index=False)

print("=" * 80)
print("Saved results to results/results_baselines.csv")
print(results_df)