# Financial News Sentiment Analysis — Stage 1: TF-IDF + Logistic Regression Baseline

This document covers the first stage of a three-model comparison project (TF-IDF + Logistic Regression → pre-trained FinBERT → fine-tuned DistilBERT) for 3-class sentiment classification (negative / neutral / positive) on financial news text. It records the data preparation pipeline, modelling decisions, and results for this baseline stage, along with the reasoning behind each choice.

## 1. Datasets

Two sources were combined to build a larger, more diverse training set than either alone:

- **FinancialPhraseBank (FPB)** — `takala/financial_phrasebank`, `sentences_allagree` config (2,264 sentences, 100% annotator agreement). Analyst-style financial news sentences.
- **FiQA Sentiment** — `TheFinAI/fiqa-sentiment-classification` (1,173 rows across its original train/test/valid splits). Financial news headlines and microblog/StockTwits-style text, originally annotated for aspect-based sentiment with a continuous score.

Combining these two also gives the project a secondary benefit: it introduces a stylistic mismatch (formal analyst sentences vs. informal headlines/microblogs) that is deliberately used later for error analysis, particularly when comparing performance against FinBERT and DistilBERT on headline-style text.

### 1.1 FPB loading issue

`datasets>=4.0` dropped support for script-based dataset loaders, and the `takala/financial_phrasebank` repo's `main` branch still carries a `.py` loading script (a mismatch left over from a partial Parquet migration). This was resolved by pinning to a specific commit where the repo had proper Parquet config folders:

```python
dataset_financial_phrasebank = load_dataset(
    "takala/financial_phrasebank",
    "sentences_allagree",
    revision="0dd3028d70cbd18ded8887e65e83343b03a50482"
)
```

**Justification:** pinning to a known-good commit avoids depending on `main`'s current (broken) state, and is more reproducible than hoping the repo gets fixed. A longer-term fix — vendoring the Parquet file into the project's own repo — was noted as worth doing before final submission, so the pipeline isn't dependent on an external repo's history staying intact.

## 2. Label Alignment

- FPB's `label` column is an integer-encoded `ClassLabel`: `0 = negative`, `1 = neutral`, `2 = positive` (confirmed against the dataset's own feature schema, not inferred).
- FiQA's `score` column is continuous, defined on a theoretical **-1 to +1** scale by the original 2018 FiQA task, not something to infer empirically per sample.

### 2.1 FiQA discretisation

FiQA's continuous score was converted into the same 3-class scheme using **fixed theoretical bounds** (-1 to +1), split into equal thirds, rather than the empirical min/max of any particular sample:

```python
def score_to_label(score):
    if score <= -1/3:
        return "negative"
    elif score <= 1/3:
        return "neutral"
    else:
        return "positive"
```

**Justification:** using the sample's empirical min/max to set bin edges would make the discretisation dependent on which extreme values happened to land in a particular split (train vs. test vs. valid) — an avoidable source of instability and sample-derived bias. The theoretical range is fixed by the annotation scheme itself, so it produces the same bin edges regardless of which subset of the data is examined.

Both datasets' labels were then aligned onto the same integer scheme (`0/1/2`) matching FPB's existing encoding.

## 3. Merging and Re-splitting FiQA

FiQA originally ships with its own train/test/valid splits (822/234/117), but these were built for a **regression** task (predicting a continuous score), not for 3-class classification, and are not stratified by the discretised label scheme.

**Decision:** all three original FiQA splits were merged together before combining with FPB, and a single fresh stratified split was performed afterward across the full combined dataset.

**Justification:** preserving FiQA's original splits would provide no benefit once the task changes from regression to classification, and keeping them separate would risk uneven class representation across FiQA's own train/test/valid relative to the merged, stratified split actually used for modelling.

## 4. Deduplication

After merging, 67 duplicate `sentence` values were found. These were checked for label consistency:

```python
dupe_check = dupe_sentences.groupby("sentence")["label"].nunique()
```

Result: 52 sentences had **consistent** labels across their duplicate rows (harmless repetition); 10 sentences had **conflicting** labels (the same sentence associated with more than one sentiment).

**Decision:** the 10 conflicting-label sentences were dropped entirely; the 52 consistent duplicates were reduced to a single row each (`drop_duplicates`).

**Justification:** the conflicting cases stem from FiQA's aspect-level annotation structure — the same sentence can carry different scores depending on which financial entity/aspect was being scored. Since this project treats sentiment classification at the **sentence level**, not the aspect level, these 10 cases represent genuine label ambiguity for the task as framed here, rather than noise that a smarter labelling rule could resolve. Dropping them (rather than e.g. majority-voting) was chosen because the sample loss is negligible (10 of 3,437 rows) and it avoids introducing an additional, harder-to-justify labelling rule for a 0.3% slice of the data.

Final deduplicated dataset: **`df`, 3,360 rows.**

## 5. Class Balance

```
label
1 (neutral)   1,857  →  ~54% (post-split: ~53–55% across train/test)
2 (positive)  1,039  →  ~30%
0 (negative)    541  →  ~15–17%
```

Roughly a 3:2:1 ratio — a moderate imbalance, not severe enough to require resampling techniques (e.g. SMOTE was considered and deliberately rejected — see below), but enough that accuracy alone would be a misleading headline metric.

## 6. Train/Test Split

**Decision:** an 80/20 stratified split (by `label`) directly on `df`, with hyperparameter selection handled via 5-fold cross-validation on the training set rather than a separate fixed validation set.

```python
train_df, test_df = train_test_split(
    df, test_size=0.2, stratify=df["label"], random_state=42
)
```

**Justification for stratification:** with the smallest class (`negative`) at only ~541 examples overall, a non-stratified random split risks meaningfully different class proportions landing in train vs. test purely by chance — a variable worth eliminating given it's essentially free to control for.

**Justification for dropping a separate fixed validation set in favour of CV:** an earlier version of this pipeline used a fixed 70/15/15 train/val/test split with `PredefinedSplit` to pin `GridSearchCV` to that exact validation fold. This was simplified to a plain 80/20 train/test split with standard 5-fold CV during hyperparameter search, because:
- 5-fold CV averages performance across 5 different train/validation partitions rather than relying on the luck of one fixed split, which is a **more robust** way to select hyperparameters, not merely a simpler one.
- It removes a validation set that would otherwise sit unused once CV was introduced, simplifying the pipeline without losing rigor.
- `test_df` remains held out and untouched throughout — cross-validation only ever sees training-set rows during hyperparameter search; it is not a substitute for the final, single, honest evaluation that `test_df` provides.

Test set size: 672 rows (20% of 3,360).

## 7. Model Pipeline

```python
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(stop_words="english")),
    ("clf", LogisticRegression(class_weight="balanced", max_iter=1000))
])
```

### 7.1 Handling class imbalance

`class_weight="balanced"` was fixed (not tuned) on the classifier, reweighting the loss inversely proportional to class frequency so that errors on the minority `negative` class count more heavily during training.

**Justification for not tuning it:** there is already a principled reason to want it enabled given the known imbalance — treating it as a fixed methodological decision rather than a hyperparameter to search over is clearer to defend and avoids diluting the tuning budget on something not genuinely in question.

**Why not resampling (e.g. SMOTE):** with ~541 negative examples in the full dataset, synthetic oversampling on sparse TF-IDF vectors tends to duplicate near-identical text patterns rather than introduce genuine variety, adding pipeline complexity for likely marginal benefit. Class weighting was judged the simpler, more defensible choice for this scale of imbalance.

## 8. Hyperparameter Tuning

### 8.1 Parameters selected — and why

Each parameter included was chosen because it controls a genuinely distinct aspect of the model, not because "there was room" in the grid:

| Parameter | Component | What it controls | Why included |
|---|---|---|---|
| `ngram_range` | Vectorizer | Whether features are single words only, or also include word pairs (e.g. capturing negation like "not good") | Changes what signal the model can see at all — a structural choice, not a fine-tune |
| `min_df` | Vectorizer | Filters out terms appearing in very few documents | Addresses noise specific to merging two different-style text sources (rare tickers, one-off phrasing) — a job `C` cannot do, since `C` only re-weights existing features, it can't remove noisy ones from the feature space |
| `C` | Classifier | Inverse regularisation strength — how much the model is allowed to fit the training data closely vs. staying conservative | The primary lever for controlling overfitting on the classifier side |

### 8.2 Parameters deliberately excluded

- **`max_features`** — initially considered, but dropped once it was recognised as overlapping with `C`'s job (both are, in different ways, ways of controlling model complexity/overfitting). With L2-regularised logistic regression, `C` alone can suppress the influence of a large vocabulary without needing a hard cap on vocabulary size first. Fixed instead at a generous value so it isn't meaningfully constraining anything.
- **`max_df`, `stop_words` (as a tunable), `sublinear_tf`** — judged low-leverage for this dataset size (2,352–2,688 training rows); fixed at sensible defaults (`stop_words="english"`) rather than searched, to keep the grid focused on parameters with a clear, distinct effect.
- **`penalty` / `solver`** — left at defaults (`l2` / `lbfgs`), since testing `l1` was considered but not prioritised for this baseline; noted as a possible future extension if interpretability of feature weights becomes a focus during error analysis.

### 8.3 Final grid

```python
param_grid = {
    "tfidf__ngram_range": [(1, 1), (1, 2)],
    "tfidf__min_df": [1, 2, 3],
    "clf__C": [0.1, 1, 10],
}
```
18 combinations, evaluated via 5-fold cross-validation on the training set, scored on **macro F1** (see Section 9 for why macro F1 rather than accuracy).

### 8.4 Result

```
Best parameters: {'clf__C': 10, 'tfidf__min_df': 1, 'tfidf__ngram_range': (1, 2)}
```

## 9. Evaluation Metric Choice: Macro F1

Given the ~3:2:1 class imbalance, **accuracy alone is misleading** — a model that always predicts `neutral` would score roughly 54% without learning anything useful.

**Macro F1** was adopted as the primary metric because it averages per-class F1 scores with **equal weight**, regardless of class size — unlike micro F1 (mathematically equivalent to accuracy for this kind of problem) or weighted F1 (which still lets the large `neutral` class dominate the average). This means a model that performs well only on `neutral` while underperforming on the minority `negative` class cannot hide that weakness behind a high macro F1 score.

## 10. Baseline Results (Test Set, n=672)

```
Accuracy: 0.781
Macro F1: 0.731

              precision    recall  f1-score   support
    negative       0.70      0.54      0.61       102
     neutral       0.84      0.86      0.85       361
    positive       0.71      0.76      0.73       209

Confusion matrix:
[[ 55  21  26]
 [ 12 312  37]
 [ 12  39 158]]

Inference time (672 examples): 0.0073s
```

### 10.1 Reading the results

- **`neutral`** performs best (F1 = 0.85), consistent with it being the largest, best-represented class.
- **`negative`** is the weakest class (F1 = 0.61, recall = 0.54) — the model misses nearly half of true negative examples. Of 102 true negatives, 26 were misclassified as `positive` — a more concerning error than confusion with `neutral`, since it reverses sentiment polarity entirely rather than just softening it.
- **`positive`**'s confusion is milder — its errors lean toward `neutral` (39 of 209) rather than toward `negative`.

### 10.2 Working hypothesis for error analysis

The negative→positive confusion is a strong candidate for deeper error analysis: financial "negative" sentiment is often expressed through hedged, indirect language (e.g. "lower-than-expected growth") rather than overtly negative vocabulary. A bag-of-words model like TF-IDF has no real sense of context or negation scope beyond what bigrams capture, which may explain this specific weakness. This is a useful, honest baseline limitation to carry forward as a comparison point once FinBERT and DistilBERT (both context-aware, transformer-based models) are evaluated on the same test set.

## 11. Next Steps

- Log this run's parameters, metrics, and model artefact to MLflow to establish the tracking pattern that will carry through FinBERT and fine-tuned DistilBERT.
- Evaluate pre-trained FinBERT out-of-the-box on the same test set, explicitly noting in the write-up that FinBERT (`ProsusAI/finbert`) was itself trained on FinancialPhraseBank, so this is not a strictly zero-shot evaluation.
- Fine-tune DistilBERT with a weighted loss to address the same class imbalance handled here via `class_weight="balanced"`.
- Carry forward the negative→positive misclassification pattern identified here as a specific point of comparison across all three models.
