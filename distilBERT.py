import torch
from datasets import Dataset
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer

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

print(train_dataset)
print(train_dataset[0].keys())

