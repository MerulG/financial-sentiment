import time

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline

import data
from sklearn.linear_model import LogisticRegression

print("loading data...")
df = data.preprocess()
train_df, test_df = train_test_split(df,test_size=0.2,random_state=42)

grid = {
    "tfidf__ngram_range": [(1, 1), (1, 2)],
    "tfidf__min_df": [1, 2, 3],
    "clf__C": [0.1, 1, 10],
}

pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(stop_words="english")),
    ("clf", LogisticRegression(class_weight="balanced", max_iter=1000))
])


grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=grid,
    cv=5,  # standard 5-fold cross-validation
    scoring="f1_macro",
    n_jobs=-1
)

print("training...")
grid_search.fit(train_df["sentence"], train_df["label"])
model = grid_search.best_estimator_
print("Best parameters:", grid_search.best_params_)

print("getting predictions...")
start = time.time()
predictions = model.predict(test_df["sentence"])
elapsed = time.time() - start
print(f"Inference time for {len(test_df)} examples: {elapsed:.4f}s")

accuracy = accuracy_score(test_df["label"], predictions)
macro_f1 = f1_score(test_df["label"], predictions, average="macro")

print("Accuracy:", accuracy)

#macro used to avergae f1 scores equally (divide by 3)
print("Macro F1:", macro_f1)

print(classification_report(test_df["label"], predictions, target_names=["negative", "neutral", "positive"]))

cm = confusion_matrix(test_df["label"], predictions)
print(cm)
