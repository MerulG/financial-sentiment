import mlflow
import torch
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix, \
    precision_recall_fscore_support
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

sentences = test_df["sentence"].tolist()
true_labels = test_df["label"].tolist()

BATCH_SIZE = 16
all_preds = []

for i in range(0, len(sentences), BATCH_SIZE):
    batch = sentences[i:i + BATCH_SIZE]
    encoded = tokenizer(batch, padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)

    with torch.no_grad():
        logits = model(**encoded).logits

    batch_preds = torch.argmax(logits, dim=1).cpu().numpy()
    all_preds.extend(batch_preds)

print(len(all_preds), "predictions collected")

# remap all predictions
remapped_preds = [label_name_to_your_id[finbert_id2label[p]] for p in all_preds]

accuracy = accuracy_score(true_labels, remapped_preds)
macro_f1 = f1_score(true_labels, remapped_preds, average="macro")

print("Accuracy:", accuracy)
print("Macro F1:", macro_f1)
print(classification_report(true_labels, remapped_preds, target_names=["negative", "neutral", "positive"]))

cm = confusion_matrix(true_labels, remapped_preds)
print(cm)

mlflow.set_experiment("financial-sentiment-analysis")
with mlflow.start_run(run_name="finbert_zero_shot_reference"):

    mlflow.log_param("model_name", "ProsusAI/finbert")
    mlflow.log_param("batch_size", BATCH_SIZE)
    mlflow.log_param("max_length", 128)
    mlflow.log_param("device", str(device))
    mlflow.set_tag("caveat", "FinBERT was pretrained on FinancialPhraseBank - not a true zero-shot eval")

    mlflow.log_metric("test_accuracy", accuracy)
    mlflow.log_metric("test_macro_f1", macro_f1)

    precision, recall, f1, support = precision_recall_fscore_support(
        true_labels, remapped_preds, average=None, labels=[0, 1, 2]
    )
    label_names = ["negative", "neutral", "positive"]
    for i, name in enumerate(label_names):
        mlflow.log_metric(f"precision_{name}", precision[i])
        mlflow.log_metric(f"recall_{name}", recall[i])
        mlflow.log_metric(f"f1_{name}", f1[i])

    with open("confusion_matrix.txt", "w") as f:
        f.write(str(cm))
    mlflow.log_artifact("confusion_matrix.txt")
