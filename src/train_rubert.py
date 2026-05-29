import os
import pandas as pd
import numpy as np

from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)

os.makedirs("results", exist_ok=True)
os.makedirs("models", exist_ok=True)

model_name = "cointegrated/rubert-tiny2"

train = pd.read_csv("data/processed/train.csv")
test = pd.read_csv("data/processed/test.csv")

# Чтобы обучение на обычном ноутбуке не занимало много часов,
# используем подвыборку train.
# Для финальной версии можно увеличить n до 20000 или 30000.
train = train.sample(n=10000, random_state=42)

# Часть train используем как validation для выбора лучшей модели
train_for_model = train.sample(frac=0.9, random_state=42)
val_for_model = train.drop(train_for_model.index)

train_dataset = Dataset.from_pandas(train_for_model[["text", "label"]])
val_dataset = Dataset.from_pandas(val_for_model[["text", "label"]])
test_dataset = Dataset.from_pandas(test[["text", "label"]])

tokenizer = AutoTokenizer.from_pretrained(model_name)


def tokenize(batch):
    return tokenizer(
        batch["text"],
        truncation=True,
        max_length=128
    )


train_dataset = train_dataset.map(tokenize, batched=True)
val_dataset = val_dataset.map(tokenize, batched=True)
test_dataset = test_dataset.map(tokenize, batched=True)

model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=2
)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    accuracy = accuracy_score(labels, predictions)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        predictions,
        average="macro",
        zero_division=0
    )

    return {
        "accuracy": accuracy,
        "macro_precision": precision,
        "macro_recall": recall,
        "macro_f1": f1
    }


training_args = TrainingArguments(
    output_dir="models/rubert_tiny",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=16,
    num_train_epochs=1,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="macro_f1",
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

trainer.train()

test_results = trainer.evaluate(test_dataset)

print("Test results:")
print(test_results)

pd.DataFrame([{
    "model": "RuBERT-tiny2",
    "accuracy": test_results["eval_accuracy"],
    "macro_precision": test_results["eval_macro_precision"],
    "macro_recall": test_results["eval_macro_recall"],
    "macro_f1": test_results["eval_macro_f1"]
}]).to_csv("results/results_rubert.csv", index=False)

trainer.save_model("models/rubert_tiny")
tokenizer.save_pretrained("models/rubert_tiny")

print("Saved results to results/results_rubert.csv")