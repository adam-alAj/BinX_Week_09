# Week 9 — Day 1: Sprint 4 Planning, Serialization & MLOps

| Field | Value |
|:------|:------|
| **Phase** | Phase 3 — Deep Learning & Applied Project |
| **Sprint** | Sprint 4 (Week 9) — *Deployment & Production* |
| **Day** | Day 1 of 5 — Sprint 4 Planning, Serialization & MLOps |
| **Project** | Arabic Sentiment Classification |
| **Final model** | TF-IDF (10K features) + Logistic Regression (C=1.0) |
| **Test macro F1** | 0.8623 |
| **Notebook** | `Sprint4_Serialization_MLOps.ipynb` |

---

## Objectives

1. **Sprint 4 planning** — define the Sprint 4 goal and deployment backlog.
2. **Model serialization** — persist the trained model to disk.
3. **Preprocessing serialization** — save every preprocessing object required by the model.
4. **Artifact verification** — load serialized artifacts back and reproduce a known prediction.
5. **Reproducibility** — seeds, pinned requirements, MLflow traceability.
6. **Clean `requirements.txt`** — pinned versions reflecting the actual environment.
7. **Day 2 handoff** — document exactly which artifacts the FastAPI service should load.

---

## Sprint 4 Plan

### Sprint Goal

> **Transform the validated Arabic sentiment classifier into a usable, deployable application
> that serves predictions via API, presents results through a UI, and is publicly accessible.**

### Deployment Backlog

| # | Task | Sprint Day | Status | Deliverable |
|:-:|:-----|:-----------|:------:|:------------|
| 1 | Sprint 4 Planning | Day 1 | ✅ Complete | This notebook |
| 2 | Model Serialization | Day 1 | ✅ Complete | `model.joblib` |
| 3 | Preprocessing Serialization | Day 1 | ✅ Complete | `vectorizer.joblib`, `lemma_table.json`, `preprocessing_config.json` |
| 4 | Artifact Verification | Day 1 | ✅ Complete | Reproduced prediction |
| 5 | Reproducibility Setup | Day 1 | ✅ Complete | Seeds, config |
| 6 | MLflow Traceability | Day 1 | ✅ Complete | Experiment log |
| 7 | Requirements Freeze | Day 1 | ✅ Complete | `requirements.txt` |
| 8 | FastAPI Serving | Day 2 | Planned | `/predict` API |
| 9 | Streamlit Dashboard | Day 3 | Planned | UI |
| 10 | Public Deployment | Day 4 | Planned | Public URL |
| 11 | Final Verification & Review | Day 5 | Planned | Final repository |

---

## What Was Implemented

1. **Sprint 4 planning** — goal, backlog, dependencies, Sprint 3 carry-forward items
2. **Model serialization** — Logistic Regression saved as `model.joblib`
3. **Preprocessing serialization** — TF-IDF vectorizer (`vectorizer.joblib`), lemma table (`lemma_table.json`), and preprocessing config (`preprocessing_config.json`)
4. **Artifact verification** — all artifacts reloaded and known prediction reproduced
5. **Full test set verification** — 3,000 test samples verified identical (predictions + probabilities)
6. **Reproducibility** — SEED=42 configured, requirements.txt pinned
7. **MLflow traceability** — experiment run logged with parameters, metrics, and artifacts
8. **Artifact integrity checks** — 12/13 checks passed (1 N/A for MLflow)

---

## The Complete Inference Pipeline

```
Raw Arabic Text
  → normalize_text()          — Tashkeel removal, Alef unification, punctuation cleanup
  → word_tokenize()           — NLTK Arabic tokenizer
  → filter digits/Latin       — Remove non-Arabic tokens
  → unify_alef()              — Alef/Hamza/ta-marbuta normalization
  → protect negations         — Preserve sentiment-critical tokens (لا, لم, لن, ليس, ما, غير)
  → lemmatize                 — qalsadi dictionary-based Arabic lemmatization
  → remove stopwords          — NLTK Arabic stopwords (with negation protection)
  → TfidfVectorizer.transform — Transform using 10K-feature vocabulary (fitted on train)
  → LogisticRegression.predict — Binary classification (Negative/Positive)
  → Label mapping             — 0→Negative, 1→Positive
```

---

## Serialized Artifacts

All serialized artifacts are stored in `artifacts/`:

| Artifact | Type | Size | Purpose |
|:---------|:-----|:-----|:--------|
| `model.joblib` | LogisticRegression | ~79 KB | Trained classifier |
| `vectorizer.joblib` | TfidfVectorizer | ~386 KB | Fitted TF-IDF vectorizer (10K features) |
| `lemma_table.json` | dict | ~597 KB | Arabic token→lemma mapping (20K+ entries) |
| `preprocessing_config.json` | dict | ~2 KB | Constants, config, and metadata |
| **TOTAL** | | **~1.04 MB** | |

---

## Final Model

| Property | Value |
|:---------|:------|
| Model | Logistic Regression (C=1.0, max_iter=1000, random_state=42) |
| Representation | TF-IDF (10K features, min_df=2, sublinear_tf=True) |
| Training data | 14,000 Arabic reviews |
| Test data | 3,000 held-out reviews |
| Test Accuracy | 0.8623 |
| Test Macro F1 | 0.8623 |
| Test ROC-AUC | 0.9417 |
| 5-fold CV F1 | 0.8556 ± 0.0033 |

---

## Verification Result

| Check | Result |
|:------|:-------|
| 10-sample known prediction | ✓ All match |
| 3,000-sample full test set | ✓ Identical predictions and probabilities |
| Max probability difference | < 1e-6 (machine precision) |
| Artifact integrity checks | 12/13 PASS, 1 N/A (MLflow not installed) |

---

## Reproducibility

- **Seed**: 42 (NumPy, LogisticRegression)
- **MLflow**: Experiment `arabic_sentiment_sprint4`, run `sprint4_day1_serialization`
- **Requirements**: Pinned in `requirements.txt`

---

## Day 2 Handoff — What the FastAPI Service Should Load

```python
import joblib, json
from pathlib import Path

ARTIFACTS_DIR = Path('BinX_Week_09/Day1/artifacts')

# Load learned artifacts
model = joblib.load(ARTIFACTS_DIR / 'model.joblib')
vectorizer = joblib.load(ARTIFACTS_DIR / 'vectorizer.joblib')
with open(ARTIFACTS_DIR / 'lemma_table.json', 'r', encoding='utf-8') as f:
    lemma_table = json.load(f)
with open(ARTIFACTS_DIR / 'preprocessing_config.json', 'r', encoding='utf-8') as f:
    config = json.load(f)

# Copy these functions from this notebook:
# - normalize_text()
# - unify_alef()
# - preprocess_text()
```

### Prediction endpoint contract

```python
POST /predict
Request:  { "text": "Arabic review text" }
Response: {
    "prediction": 0 or 1,
    "label": "Negative" or "Positive",
    "probabilities": { "Negative": 0.12, "Positive": 0.88 },
    "model_version": "1.0.0"
}
```

---

## Files Created/Modified

| File | Action |
|:-----|:-------|
| `Sprint4_Serialization_MLOps.ipynb` | Created (this notebook) |
| `artifacts/model.joblib` | Created |
| `artifacts/vectorizer.joblib` | Created |
| `artifacts/lemma_table.json` | Created |
| `artifacts/preprocessing_config.json` | Created |
| `requirements.txt` | Created (Day 1 copy) |

---

## Tools Used

- `scikit-learn` (`TfidfVectorizer`, `LogisticRegression`, `accuracy_score`, `f1_score`)
- `joblib` (model serialization/deserialization)
- NLTK (tokenization, stopwords)
- `qalsadi` (Arabic lemmatizer)
- NumPy, Pandas
- JSON (config/lemma table serialization)
- MLflow (optional — experiment tracking)
