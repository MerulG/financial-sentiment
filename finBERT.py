import torch
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification

import data

print("loading data...")
df = data.preprocess()
train_df, test_df = train_test_split(df,test_size=0.2,random_state=42)

MODEL_NAME = "ProsusAI/finbert"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"using device: {device}")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
model.to(device)
model.eval()

# match labels to preprocessing script
finbert_id2label = model.config.id2label
print("FinBERT's native label mapping:", finbert_id2label)

label_name_to_your_id = {"negative": 0, "neutral": 1, "positive": 2}
print("FinBERT's new label mapping:", label_name_to_your_id)



