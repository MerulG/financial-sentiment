from datasets import load_dataset

dataset_financial_phrasebank = load_dataset(
    "takala/financial_phrasebank",
    "sentences_allagree",
    revision="0dd3028d70cbd18ded8887e65e83343b03a50482"
)
dataset_fiqa = load_dataset("TheFinAI/fiqa-sentiment-classification")

df_fpb = dataset_financial_phrasebank["train"].to_pandas()

df_fiqa_train = dataset_fiqa["train"].to_pandas()
df_fiqa_test = dataset_fiqa["test"].to_pandas()
df_fiqa_valid = dataset_fiqa["valid"].to_pandas()

print(df_fpb.shape)
print(df_fiqa_train.shape)
print(df_fiqa_test.shape)
print(df_fiqa_valid.shape)

print(df_fpb.columns.tolist())
print(df_fiqa_train.columns.tolist())
