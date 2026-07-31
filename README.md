# Financial News Sentiment Analysis

A three-model comparison project for 3-class sentiment classification (negative / neutral / positive) on financial news text. Built as an ML portfolio piece demonstrating the full pipeline from classical baselines through transformer fine-tuning, with systematic experiment tracking via MLflow.

**MLflow experiment:** `financial-sentiment-analysis`

---

## Results Summary

| Model | Accuracy | Macro F1 | Inference Time (672 examples) |
|---|---|---|---|
| TF-IDF + Logistic Regression | 0.781 | 0.731 | 0.007s |
| FinBERT (zero-shot reference) | 0.814 | 0.796 | — |
| DistilBERT (fine-tuned) | **0.853** | **0.832** | 2.75s |

The negative-class recall jump from 0.54 (Stage 1) to 0.833 (Stage 3) is the headline finding: financial negative sentiment — typically expressed through hedged, indirect language — is poorly captured by bag-of-words models but responds strongly to fine-tuning on domain text.

---

## Project Structure

```
├── data/
│   └── prepare.py           # Dataset loading, merging, deduplication, splitting
├── stage1_baseline/
│   └── train.py             # TF-IDF + Logistic Regression + MLflow logging
├── stage2_finbert/
│   └── evaluate.py          # FinBERT inference + MLflow logging
├── stage3_distilbert/
│   └── train.py             # DistilBERT fine-tuning + MLflow logging
└── requirements.txt
```

---

## Dataset

Two sources were combined to build a larger, more diverse dataset:

- **FinancialPhraseBank** (`takala/financial_phrasebank`, `sentences_allagree` config) — 2,264 sentences from analyst-style financial news, annotated by finance professionals at 100% inter-annotator agreement.
- **FiQA Sentiment** (`TheFinAI/fiqa-sentiment-classification`) — 1,173 examples from financial headlines and microblog-style text (StockTwits, news), originally annotated with continuous sentiment scores.

### Combining the datasets

FiQA uses continuous scores (-1 to +1); FinancialPhraseBank uses discrete 3-class labels. FiQA scores were discretised using fixed theoretical bounds rather than empirical min/max, to avoid data leakage and ensure thresholds are interpretable regardless of sample:

```
score > 0.333  → positive
score < -0.333 → negative
otherwise      → neutral
```

After label alignment, merging, and deduplication (67 duplicates found — 52 with consistent labels retained, 10 with conflicting labels dropped entirely), the final dataset was **3,360 rows**.

An 80/20 stratified train/test split was applied (`random_state=42`), giving **2,688 training** and **672 test** examples. The same split is used identically across all three stages so the test set is a fair comparison surface.

### Class distribution

The dataset has a ~3:2:1 imbalance (neutral dominant, negative minority). This informed the choice of macro F1 as the primary comparison metric throughout — it weights all three classes equally, preventing a model that ignores the negative minority from appearing stronger than it is.

---

## Stage 1 — TF-IDF + Logistic Regression (Baseline)

**Goal:** Establish a classical NLP baseline as the performance floor for the comparison.

### Approach

A Scikit-learn `Pipeline` combining `TfidfVectorizer` and `LogisticRegression(class_weight="balanced")`. Hyperparameters tuned via `GridSearchCV` with 5-fold cross-validation on the training set, scored on macro F1. 18 combinations tested across `ngram_range`, `min_df`, and regularisation `C`.

Parameters deliberately excluded from the grid with justification:
- `max_features` — redundant with `C` for controlling overfitting under L2 regularisation
- `stop_words` — low-leverage for this dataset size and domain
- `sublinear_tf` — fixed at default; not expected to move the needle at 3K examples

### Results

**Best params:** `C=10, min_df=1, ngram_range=(1,2)`

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| negative | — | 0.54 | 0.61 | 102 |
| neutral | — | 0.86 | 0.85 | 361 |
| positive | — | 0.76 | 0.73 | 209 |
| **Accuracy** | | | **0.781** | |
| **Macro F1** | | | **0.731** | |

**Key weakness:** Negative-class recall of 0.54 — the model misses nearly half of true negative examples. Of 102 true negatives, 26 were misclassified as positive. Financial negative sentiment is often expressed through hedged language ("lower-than-expected growth") rather than overtly negative vocabulary, which a bag-of-words model cannot capture beyond what bigrams encode.

---

## Stage 2 — FinBERT (State-of-the-Art Reference)

**Goal:** Establish a domain-pretrained transformer as a reference point, without any fine-tuning on this dataset.

### Approach

`ProsusAI/finbert` loaded via HuggingFace and evaluated directly on the test set. No training performed.

**Important caveat:** FinBERT was itself pretrained on FinancialPhraseBank — one of the two datasets used here. This means Stage 2 is not a true zero-shot evaluation; FinBERT has already seen text from the same distribution. This is acknowledged explicitly in the MLflow run params. The comparison remains useful as a reference point for what a domain-pretrained model achieves, but results should not be interpreted as zero-shot generalisation.

A non-trivial implementation detail: FinBERT's native `id2label` mapping (`{0: positive, 1: negative, 2: neutral}`) differs from the label scheme used in this project (`{0: negative, 1: neutral, 2: positive}`). Labels were remapped via a string intermediary to avoid silent misalignment.

### Results

| Class | Recall (notable) |
|---|---|
| positive | **0.72** (weakest area) |

| Metric | Value |
|---|---|
| Accuracy | **0.814** |
| Macro F1 | **0.796** |

FinBERT's weakest area is positive-class recall (0.72) — a different failure mode than the TF-IDF baseline, which struggled with negative. Domain pretraining shifts which classes are harder, but does not eliminate class-specific weaknesses.

---

## Stage 3 — DistilBERT Fine-Tuned

**Goal:** Fine-tune a general-purpose transformer on this dataset — the primary ML deliverable of the project.

### Approach

`distilbert-base-uncased` loaded via HuggingFace `AutoModelForSequenceClassification` with `num_labels=3`. Fine-tuned using the HuggingFace `Trainer` API.

Key training decisions:

| Decision | Value | Reason |
|---|---|---|
| `max_length` | 128 | Sufficient for financial sentences; longer sequences add memory cost with minimal gain |
| `learning_rate` | 2e-5 | Standard for transformer fine-tuning; preserves pretrained representations |
| `num_train_epochs` | 4 | Balances convergence against overfitting on a ~2.7K training set |
| `batch_size` | 16 | Fits available memory |
| `save_strategy` | `"no"` | No validation set; checkpoint selection would require peeking at test data |
| `report_to` | `"none"` | Manual MLflow logging kept explicit and matching across all three stages |

### Results

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| negative | 0.727 | **0.833** | 0.776 | 102 |
| neutral | 0.886 | 0.886 | 0.886 | 361 |
| positive | 0.866 | 0.804 | 0.834 | 209 |
| **Accuracy** | | | **0.853** | |
| **Macro F1** | | | **0.832** | |
| Inference time | | | 2.75s (672 examples) | |

---

## Three-Way Comparison and Error Analysis

### Headline numbers

| Model | Accuracy | Macro F1 | Negative Recall |
|---|---|---|---|
| TF-IDF + LogReg | 0.781 | 0.731 | 0.54 |
| FinBERT (zero-shot-ish) | 0.814 | 0.796 | — |
| DistilBERT (fine-tuned) | **0.853** | **0.832** | **0.833** |

### Key findings

**The negative-class recall jump is the most striking result.** TF-IDF flagged recall of 0.54 on the negative class as its primary weakness. Fine-tuned DistilBERT — despite starting with no domain-specific pretraining, unlike FinBERT — achieves 0.833 negative recall. Financial negative sentiment relies on contextual cues (hedging, indirection, negation scope) that a bag-of-words model cannot capture, but that a fine-tuned transformer learns directly from training examples.

**FinBERT's failure mode differs from the baseline's.** Where TF-IDF struggled most with negative sentiment, FinBERT's weakest area was positive-class recall (0.72). Domain pretraining shifts the model's weaknesses rather than eliminating them — it does not substitute for task-specific fine-tuning.

**The FinBERT caveat matters.** FinBERT's stronger-than-expected performance is at least partly explained by pretraining on FinancialPhraseBank. A true zero-shot comparison would require a held-out dataset from a different source; that limitation is acknowledged here rather than papered over.

**Inference cost is the real tradeoff.** TF-IDF classifies 672 examples in 7ms. DistilBERT takes 2.75 seconds — a 400× difference. For a production system classifying thousands of headlines in real time, this gap is material. The accuracy gain (0.781 → 0.853) needs to be weighed against latency and serving cost in any real deployment context.

---

## Stack

| Area | Tools |
|---|---|
| Classical ML | Scikit-learn, NumPy, Pandas |
| Transformers | HuggingFace Transformers, PyTorch |
| Experiment tracking | MLflow |
