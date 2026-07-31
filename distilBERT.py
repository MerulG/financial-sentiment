import time

import mlflow
import numpy as np
import torch
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer

import data

print("loading data...")
df = data.preprocess()
train_df, test_df = train_test_split(df,test_size=0.2,random_state=42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"using device: {device}")

MODEL_NAME = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Sample
sample_text = train_df["sentence"].iloc[0]
sample_encoding = tokenizer(sample_text, truncation=True, padding="max_length", max_length=128)

print("Original text:", sample_text)
print("Token IDs (first 20):", sample_encoding["input_ids"][:20])
print("Attention mask (first 20):", sample_encoding["attention_mask"][:20])
print("Decoded back:", tokenizer.decode(sample_encoding["input_ids"]))

# Convert existing pandas train/test DataFrames into HF Dataset objects.
train_dataset = Dataset.from_pandas(train_df[["sentence", "label"]].reset_index(drop=True))
test_dataset = Dataset.from_pandas(test_df[["sentence", "label"]].reset_index(drop=True))

def tokenize_function(examples):
    return tokenizer(examples["sentence"], truncation=True, padding="max_length", max_length=128,)

train_dataset = train_dataset.map(tokenize_function, batched=True)
test_dataset = test_dataset.map(tokenize_function, batched=True)

# Trainer expected 'labels'
train_dataset = train_dataset.rename_column("label", "labels")
test_dataset = test_dataset.rename_column("label", "labels")

train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
test_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

id2label = {0: "negative", 1: "neutral", 2: "positive"}
label2id = {"negative": 0, "neutral": 1, "positive": 2}

model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=3, id2label=id2label, label2id=label2id,)
model.to(device)

training_args = TrainingArguments(
    output_dir="./distilbert-finetuned",
    num_train_epochs=4,
    per_device_train_batch_size=16,
    learning_rate=2e-5,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=50,
    save_strategy="no",
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
)

start_time = time.time()
train_result = trainer.train()
training_time = time.time() - start_time

print(f"Training completed in {training_time:.1f} seconds")
print(train_result)

start_time = time.time()
predictions_output = trainer.predict(test_dataset)
inference_time_seconds = time.time() - start_time

logits = predictions_output.predictions
y_pred = np.argmax(logits, axis=1)
y_true = predictions_output.label_ids

test_accuracy = accuracy_score(y_true, y_pred)
test_macro_f1 = f1_score(y_true, y_pred, average="macro")

class_names = ["negative", "neutral", "positive"]
precision, recall, f1, support = precision_recall_fscore_support(
    y_true, y_pred, labels=[0, 1, 2], average=None
)

print(f"test_accuracy: {test_accuracy:.4f}")
print(f"test_macro_f1: {test_macro_f1:.4f}")
print()
for i, name in enumerate(class_names):
    print(f"{name}: precision={precision[i]:.4f}, recall={recall[i]:.4f}, f1={f1[i]:.4f}, support={support[i]}")

mlflow.set_experiment("financial-sentiment-analysis")

with mlflow.start_run(run_name="distilbert-finetuned"):
    # Params — mirrors the level of detail you logged for stages 1 and 2
    mlflow.log_param("model", "distilbert-base-uncased")
    mlflow.log_param("max_length", 128)
    mlflow.log_param("learning_rate", 2e-5)
    mlflow.log_param("num_train_epochs", 4)
    mlflow.log_param("batch_size", 16)
    mlflow.log_param("train_samples", len(train_dataset))
    mlflow.log_param("test_samples", len(test_dataset))

    # Headline metrics — exact same names as stages 1 & 2
    mlflow.log_metric("test_accuracy", test_accuracy)
    mlflow.log_metric("test_macro_f1", test_macro_f1)
    mlflow.log_metric("inference_time_seconds", inference_time_seconds)

    for i, name in enumerate(class_names):
        mlflow.log_metric(f"precision_{name}", precision[i])
        mlflow.log_metric(f"recall_{name}", recall[i])
        mlflow.log_metric(f"f1_{name}", f1[i])

    mlflow.log_metric("training_time_seconds", training_time)

print("Logged to MLflow experiment: financial-sentiment-analysis")