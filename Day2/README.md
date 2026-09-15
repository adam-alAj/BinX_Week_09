# Week 9 — Day 2: Serving the Model with FastAPI

| Field | Value |
|:------|:------|
| **Phase** | Phase 3 — Deep Learning & Applied Project |
| **Sprint** | Sprint 4 (Week 9) — *Deployment & Production* |
| **Day** | Day 2 of 5 — Serving the Model with FastAPI |
| **Project** | Arabic Sentiment Classification |
| **Final model** | TF-IDF (10K features) + Logistic Regression (C=1.0) |
| **Test macro F1** | 0.8623 |
| **Notebook** | `FastAPI.ipynb` |
| **API App** | `main.py` |

---

## Objectives

1. **FastAPI application** — local model-serving REST API.
2. **Pydantic validation** — request schema enforcement with `field_validator`.
3. **Preprocessing reuse** — exact same pipeline as training (shared `preprocessing.py` module).
4. **Serialized artifact loading** — model + vectorizer + lemma table + config from Day 1.
5. **`/predict` POST endpoint** — raw Arabic text → JSON prediction response.
6. **Error handling** — graceful responses for invalid inputs.
7. **`/docs` testing** — Swagger UI walkthrough for API exploration.
8. **Notebook vs API verification** — same input, same output (consistency check).
9. **Day 3 handoff** — document Streamlit integration requirements.

---

## What Was Implemented

1. **FastAPI application** (`main.py`) with three endpoints:
   - `GET /` — health check confirming model is loaded
   - `GET /health` — lightweight health probe
   - `POST /predict` — sentiment classification with confidence scores
2. **Pydantic request/response schemas** — `PredictionRequest` with empty-text validation, `PredictionResponse` with prediction, label, confidence, probabilities, model version, and preprocessing step count.
3. **Shared preprocessing module** (`preprocessing.py`) — reusable `preprocess_text()` and `load_artifacts()` functions, imported by both the FastAPI app and the Day 3 Streamlit dashboard.
4. **Artifact loading at startup** — all Day 1 artifacts (`model.joblib`, `vectorizer.joblib`, `lemma_table.json`, `preprocessing_config.json`) loaded once on server start, cached in a module-level variable.
5. **Full inference pipeline** — raw text → `preprocess_text()` → `vectorizer.transform()` → `model.predict()` → JSON response.
6. **Notebook-vs-API consistency** — 6 Arabic test samples verified identical predictions and probabilities between the notebook pipeline and the FastAPI `/predict` endpoint (10/10 match).
7. **Swagger UI documentation** — auto-generated API docs at `/docs` with request/response examples.

---

## Architecture

```
Client Request (POST /predict)
    ↓
Pydantic Validation (PredictionRequest)
    ↓
preprocessing.py (shared module from training)
    ↓
Loaded TF-IDF vectorizer (Day 1 artifact)
    ↓
Loaded Logistic Regression model (Day 1 artifact)
    ↓
Prediction + Probabilities
    ↓
JSON Response (PredictionResponse)
```

---

## API Endpoints

| Method | Path | Description |
|:-------|:-----|:------------|
| `GET` | `/` | Health check — confirms model is loaded |
| `GET` | `/health` | Lightweight health probe |
| `POST` | `/predict` | Classify sentiment of Arabic review text |

### POST /predict

**Request:**
```json
{
  "text": "هذا المنتج ممتاز جدا"
}
```

**Response:**
```json
{
  "prediction": 1,
  "label": "Positive (1)",
  "confidence": 0.9761,
  "probabilities": {
    "Negative (0)": 0.0239,
    "Positive (1)": 0.9761
  },
  "model_version": "1.0.0",
  "preprocessing_steps": 7
}
```

---

## Preprocessing Pipeline (Shared Module)

```
Raw Arabic Text
  → normalize_text()          — Tashkeel removal, Alef unification, punctuation cleanup
  → word_tokenize()           — NLTK Arabic tokenizer
  → filter digits/Latin       — Remove non-Arabic tokens
  → unify_alef()              — Alef/Hamza/ta-marbuta normalization
  → protect negations         — Preserve sentiment-critical tokens
  → lemmatize                 — qalsadi dictionary-based Arabic lemmatization
  → remove stopwords          — NLTK Arabic stopwords (with negation protection)
  → TfidfVectorizer.transform — Transform using 10K-feature vocabulary
  → LogisticRegression.predict — Binary classification (Negative/Positive)
```

---

## Files Created/Modified

| File | Purpose |
|:-----|:--------|
| `FastAPI.ipynb` | Documentation notebook with implementation walkthrough and verification |
| `main.py` | FastAPI application with `/predict` endpoint |
| `preprocessing.py` | Shared preprocessing module (reused by Day 3 Streamlit dashboard) |

---

## How to Run

```bash
# From the BinX_Week_09/Day2/ directory:
uvicorn main:app --reload

# API docs available at:
# http://127.0.0.1:8000/docs
```

---

## Tools Used

- **FastAPI** — async web framework for the REST API
- **Pydantic** — request/response validation and serialization
- **Uvicorn** — ASGI server for local deployment
- **scikit-learn** — `TfidfVectorizer`, `LogisticRegression` (loaded from artifacts)
- **joblib** — model deserialization
- **NLTK** — tokenization, stopwords
- **qalsadi** — Arabic lemmatizer
- **NumPy**, **Pandas**, **JSON**
