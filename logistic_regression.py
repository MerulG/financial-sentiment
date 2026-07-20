import time

import mlflow
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix, \
    precision_recall_fscore_support
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

mlflow.set_experiment("financial-sentiment-analysis")
with mlflow.start_run(run_name="tfidf_logreg_baseline"):

    print("training...")
    grid_search.fit(train_df["sentence"], train_df["label"])
    model = grid_search.best_estimator_
    print("Best parameters:", grid_search.best_params_)

    # Log winning hyperparameters
    best_params = grid_search.best_params_
    mlflow.log_param("ngram_range", best_params["tfidf__ngram_range"])
    mlflow.log_param("min_df", best_params["tfidf__min_df"])
    mlflow.log_param("C", best_params["clf__C"])
    mlflow.log_param("class_weight", "balanced")
    mlflow.log_param("stop_words", "english")
    mlflow.log_metric("cv_macro_f1", grid_search.best_score_)

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

    # Log test metrics
    mlflow.log_metric("test_accuracy", accuracy)
    mlflow.log_metric("test_macro_f1", macro_f1)
    mlflow.log_metric("inference_time_seconds", elapsed)

    # compute metrics for each class separately
    precision, recall, f1, support = precision_recall_fscore_support(
        test_df["label"], predictions, average=None, labels=[0, 1, 2]
    )
    label_names = ["negative", "neutral", "positive"]
    for i, name in enumerate(label_names):
        mlflow.log_metric(f"precision_{name}", precision[i])
        mlflow.log_metric(f"recall_{name}", recall[i])
        mlflow.log_metric(f"f1_{name}", f1[i])

    # Log confusion matrix as an artifact
    with open("confusion_matrix.txt", "w") as f:
        f.write(str(cm))
    mlflow.log_artifact("confusion_matrix.txt")

    # Log the fitted pipeline itself
    mlflow.sklearn.log_model(model, "model")
