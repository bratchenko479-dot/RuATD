import os
import pandas as pd

from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report

os.makedirs("results", exist_ok=True)

train = pd.read_csv("data/processed/train.csv")
test = pd.read_csv("data/processed/test.csv")

x_train = train["text"]
y_train = train["label"]

x_test = test["text"]
y_test = test["label"]


def evaluate_model(name, model, save_predictions=False):
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
    print(classification_report(y_test, predictions, zero_division=0))

    if save_predictions:
        pred_df = test.copy()
        pred_df["prediction"] = predictions
        pred_df.to_csv("results/best_model_predictions.csv", index=False)

        errors = pred_df[pred_df["label"] != pred_df["prediction"]].copy()
        errors.to_csv("results/errors_best_model.csv", index=False)

        false_positives = pred_df[
            (pred_df["label"] == 0) & (pred_df["prediction"] == 1)
        ].copy()
        false_negatives = pred_df[
            (pred_df["label"] == 1) & (pred_df["prediction"] == 0)
        ].copy()

        false_positives.to_csv("results/false_positives.csv", index=False)
        false_negatives.to_csv("results/false_negatives.csv", index=False)

    return {
        "model": name,
        "accuracy": accuracy,
        "macro_precision": precision,
        "macro_recall": recall,
        "macro_f1": f1
    }


results = []

# 1. Majority baseline
majority_model = DummyClassifier(strategy="most_frequent")

results.append(evaluate_model("Majority baseline", majority_model))


# 2. Word TF-IDF + Logistic Regression
word_logreg_model = Pipeline([
    ("tfidf", TfidfVectorizer(
        analyzer="word",
        lowercase=True,
        max_features=50000,
        ngram_range=(1, 2)
    )),
    ("classifier", LogisticRegression(
        max_iter=1000,
        class_weight="balanced"
    ))
])

results.append(evaluate_model("Word TF-IDF + Logistic Regression", word_logreg_model))


# 3. Word TF-IDF + Linear SVM
word_svm_model = Pipeline([
    ("tfidf", TfidfVectorizer(
        analyzer="word",
        lowercase=True,
        max_features=50000,
        ngram_range=(1, 2)
    )),
    ("classifier", LinearSVC(
        class_weight="balanced"
    ))
])

results.append(evaluate_model("Word TF-IDF + Linear SVM", word_svm_model))


# 4. Character TF-IDF + Logistic Regression
char_logreg_model = Pipeline([
    ("tfidf", TfidfVectorizer(
        analyzer="char",
        lowercase=True,
        max_features=100000,
        ngram_range=(3, 5)
    )),
    ("classifier", LogisticRegression(
        max_iter=1000,
        class_weight="balanced"
    ))
])

results.append(evaluate_model("Char TF-IDF + Logistic Regression", char_logreg_model))


# 5. Character TF-IDF + Linear SVM
char_svm_model = Pipeline([
    ("tfidf", TfidfVectorizer(
        analyzer="char",
        lowercase=True,
        max_features=100000,
        ngram_range=(3, 5)
    )),
    ("classifier", LinearSVC(
        class_weight="balanced"
    ))
])

results.append(evaluate_model("Char TF-IDF + Linear SVM", char_svm_model))


# 6. Word + Char TF-IDF + Logistic Regression
word_char_logreg_model = Pipeline([
    ("features", FeatureUnion([
        ("word_tfidf", TfidfVectorizer(
            analyzer="word",
            lowercase=True,
            max_features=50000,
            ngram_range=(1, 2)
        )),
        ("char_tfidf", TfidfVectorizer(
            analyzer="char",
            lowercase=True,
            max_features=100000,
            ngram_range=(3, 5)
        ))
    ])),
    ("classifier", LogisticRegression(
        max_iter=1000,
        class_weight="balanced"
    ))
])

# Для error analysis сохраняем предсказания этой модели.
# Если потом другая модель окажется лучше, поменяем save_predictions=True на неё.
results.append(evaluate_model(
    "Word+Char TF-IDF + Logistic Regression",
    word_char_logreg_model,
    save_predictions=True
))


results_df = pd.DataFrame(results)
results_df = results_df.sort_values("macro_f1", ascending=False)
results_df.to_csv("results/results_baselines.csv", index=False)

print("=" * 80)
print("Saved results to results/results_baselines.csv")
print(results_df)