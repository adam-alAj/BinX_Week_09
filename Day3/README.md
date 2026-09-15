# Week 9 — Day 3: Interactive Streamlit Dashboard

| Field | Value |
|:------|:------|
| **Phase** | Phase 3 — Deep Learning & Applied Project |
| **Sprint** | Sprint 4 (Week 9) — *Deployment & Production* |
| **Day** | Day 3 of 5 — Interactive Streamlit Dashboard |
| **Project** | Arabic Sentiment Classification |
| **Final model** | TF-IDF (10K features) + Logistic Regression (C=1.0) |
| **Test macro F1** | 0.8623 |
| **Notebook** | `Dashboard.ipynb` |
| **App** | `app.py` |

---

## Objectives

1. **Streamlit app** — serves the trained model to non-technical users.
2. **Appropriate widgets** — text area for Arabic review input with example selector.
3. **Clean demo UI** — suitable for live presentation.
4. **Prominent prediction** — clear sentiment result display with confidence.
5. **Supporting visualization** — probability distribution bar chart.
6. **Local execution** — app runs and is testable.
7. **Mentor Code Review readiness** — clean, documented code.

---

## What Was Implemented

1. **Streamlit dashboard** (`app.py`) — complete interactive web application:
   - `st.text_area` with placeholder and example selector dropdown
   - `st.button` for triggering analysis
   - `st.success` / `st.error` for prediction display with confidence
   - `st.metric` for prediction label and confidence score
   - Matplotlib horizontal bar chart of class probabilities
   - Expandable "How it works" and "Model Details" sections
   - Empty/invalid input validation before prediction
2. **Shared preprocessing reuse** — imports `preprocessing.py` from Day 2 (exact same functions as training).
3. **Cached artifact loading** — `@st.cache_resource` ensures artifacts load once, not on every rerun.
4. **Prediction consistency validation** — 6 Arabic test samples verified identical predictions between the notebook pipeline and the Streamlit app logic (6/6 match).
5. **Functional testing** — 8 Arabic test cases covering positive/negative sentiment across different domains.
6. **Error handling tests** — empty string, whitespace, Latin text, numbers-only, and None input all handled gracefully.
7. **Final validation** — 10/10 checks pass, confirming artifacts load from disk, model is correct, inference works, and no retraining occurred.

---

## Architecture

```
User Input (Streamlit text_area)
    ↓
preprocessing.py (shared module from Day 2)
    ↓
Loaded TF-IDF vectorizer (Day 1 artifact)
    ↓
Loaded Logistic Regression model (Day 1 artifact)
    ↓
Prediction + Probabilities
    ↓
Streamlit UI (result + visualization)
```

---

## Key Features

| Feature | Implementation |
|:--------|:---------------|
| Input | `st.text_area` with placeholder and example selector |
| Validation | Empty input check before prediction |
| Prediction display | `st.success` / `st.error` with confidence percentage |
| Metrics | `st.metric` for prediction label and confidence |
| Visualization | Matplotlib horizontal bar chart of class probabilities |
| Explanation | Expandable model details section |
| Error handling | Try/except with user-friendly error messages |
| Caching | `@st.cache_resource` for artifact loading (load once) |

---

## Validation Result (10/10 Checks)

| # | Check | Result |
|:-:|:------|:-------|
| 1 | Artifacts load successfully | ✅ PASS |
| 2 | Model type correct (LogisticRegression) | ✅ PASS |
| 3 | Vectorizer vocab = 10,000 | ✅ PASS |
| 4 | Lemma table loaded (20,381 entries) | ✅ PASS |
| 5 | Inference: positive input | ✅ PASS |
| 6 | Inference: negative input | ✅ PASS |
| 7 | Probabilities sum to 1.0 | ✅ PASS |
| 8 | Artifacts loaded from disk (no retraining) | ✅ PASS |
| 9 | app.py file exists | ✅ PASS |
| 10 | Prediction consistency (deterministic) | ✅ PASS |

---

## Files Created/Modified

| File | Purpose |
|:-----|:--------|
| `Dashboard.ipynb` | Documentation notebook with validation and functional tests |
| `app.py` | Streamlit dashboard application |

---

## How to Run

```bash
# From the BinX_Week_09/Day3/ directory:
streamlit run app.py

# The app will open in your browser at:
# http://localhost:8501
```

---

## Tools Used

- **Streamlit** — interactive web application framework
- **Matplotlib** — probability distribution visualization
- **scikit-learn** — `TfidfVectorizer`, `LogisticRegression` (loaded from artifacts)
- **joblib** — model deserialization
- **NLTK** — tokenization, stopwords
- **qalsadi** — Arabic lemmatizer
- **NumPy**, **JSON**
