from datasets import load_dataset
import pandas as pd
from sklearn.model_selection import train_test_split

def preprocess():
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
    df_fiqa_all = pd.concat([df_fiqa_train, df_fiqa_test, df_fiqa_valid], ignore_index=True)

    df_fiqa_all = df_fiqa_all.drop(columns=['_id','target','aspect','type'])

    df_fiqa_all["label"] = df_fiqa_all["score"].apply(score_to_label)
    df_fiqa_all = df_fiqa_all.drop(columns=['score'])

    df = pd.concat([df_fiqa_all, df_fpb], ignore_index=True)

    #delete conflicting duplicates and remove duplicates
    dupe_sentences = df[df["sentence"].duplicated(keep=False)]
    dupe_check = dupe_sentences.groupby("sentence")["label"].nunique()
    conflicting_sentences = dupe_check[dupe_check > 1].index

    df = df[~df["sentence"].isin(conflicting_sentences)]
    df = df.drop_duplicates(subset="sentence", keep="first")

    #test_df, val_df = train_test_split(test_val_df,test_size=0.5,random_state=42)
    return df


#0=negative, 1=neutral, 2=positive
def score_to_label(score):
    if score <= -1/3:
        return 0
    elif score <= 1/3:
        return 1
    else:
        return 2
