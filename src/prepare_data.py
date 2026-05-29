import os
import pandas as pd

os.makedirs("data/processed", exist_ok=True)

train = pd.read_csv("data/raw/train.csv")
val = pd.read_csv("data/raw/val.csv")
test_without_labels = pd.read_csv("data/raw/test-2.csv")

print("Original train shape:", train.shape)
print("Original val shape:", val.shape)
print("Original test shape:", test_without_labels.shape)

# Берём нужные колонки и переименовываем их в простой формат
train = train[["Id", "Text", "Class"]].copy()
val = val[["Id", "Text", "Class"]].copy()
test_without_labels = test_without_labels[["Id", "Text"]].copy()

train = train.rename(columns={
    "Id": "id",
    "Text": "text",
    "Class": "label"
})

val = val.rename(columns={
    "Id": "id",
    "Text": "text",
    "Class": "label"
})

test_without_labels = test_without_labels.rename(columns={
    "Id": "id",
    "Text": "text"
})

# Превращаем классы H и M в числа
# H = Human = 0
# M = Machine = 1
label_map = {
    "H": 0,
    "M": 1
}

train["label"] = train["label"].map(label_map)
val["label"] = val["label"].map(label_map)

# На всякий случай превращаем текст в строку
train["text"] = train["text"].astype(str)
val["text"] = val["text"].astype(str)
test_without_labels["text"] = test_without_labels["text"].astype(str)

# Проверяем, что метки нормально преобразовались
print("\nTrain labels:")
print(train["label"].value_counts())

print("\nVal labels:")
print(val["label"].value_counts())

# Сохраняем обработанные данные
train.to_csv("data/processed/train.csv", index=False)

# val используем как test, потому что в настоящем test-2.csv нет ответов
val.to_csv("data/processed/test.csv", index=False)

# настоящий test без ответов сохраняем отдельно, он пригодится позже
test_without_labels.to_csv("data/processed/test_without_labels.csv", index=False)

print("\nSaved files:")
print("data/processed/train.csv")
print("data/processed/test.csv")
print("data/processed/test_without_labels.csv")

print("\nProcessed train:")
print(train.head())

print("\nProcessed test:")
print(val.head())