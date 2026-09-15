# BinX_Week_09

## 📅 Week 9: Deep Learning & Applied Project — Sprint 4 (In Progress 🔄)

**Sprint Theme:** Deployment & Production

**Sprint Goal:** Transform the validated Arabic sentiment classifier into a usable, deployable application that serves predictions via API, presents results through a UI, and is publicly accessible.

---

### Sprint 4 Backlog

| # | Task | Sprint Day | Status |
|:-:|:-----|:-----------|:------:|
| 1 | Sprint 4 Planning | Day 1 | ✅ Complete |
| 2 | Model Serialization | Day 1 | ✅ Complete |
| 3 | Preprocessing Serialization | Day 1 | ✅ Complete |
| 4 | Artifact Verification | Day 1 | ✅ Complete |
| 5 | Reproducibility Setup | Day 1 | ✅ Complete |
| 6 | MLflow Traceability | Day 1 | ✅ Complete |
| 7 | Requirements Freeze | Day 1 | ✅ Complete |   | 8 | FastAPI Serving | Day 2 | ✅ Complete |
   | 9 | Streamlit Dashboard | Day 3 | ✅ Complete |
| 10 | Public Deployment | Day 4 | Planned |
| 11 | Final Verification & Review | Day 5 | Planned |

---

### ✅ Day 1: Sprint 4 Planning, Serialization & MLOps

* **Objective:** Completing Sprint 4 planning and transitioning the validated Sprint 3 pipeline into a production-ready state by serializing the model and all preprocessing objects, establishing reproducibility, and preparing the foundation for Day 2 FastAPI serving, Day 3 Streamlit UI, and Day 4 public deployment.
* **Key Tasks & Accomplishments:**
  - Completed **Sprint 4 planning** with a deployment backlog (11 tasks covering planning, serialization, API serving, UI, deployment, and review).
  - Defined the **Sprint 4 goal**: transform the Arabic sentiment classifier into a usable, deployable application.
  - Reviewed the Sprint 3 retrospective and identified carry-forward items for Sprint 4.
  - Rebuilt the complete inference pipeline from source artifacts (TF-IDF + Logistic Regression) and verified reproduction against Week 8 baselines (**Test Accuracy 0.8623, F1 0.8623**).
  - **Serialized the trained model** (`model.joblib`) using `joblib.dump()`.
  - **Serialized preprocessing artifacts**:
    - `vectorizer.joblib` — fitted TF-IDF vectorizer (10K features)
    - `lemma_table.json` — Arabic token→lemma mapping (20K+ entries)
    - `preprocessing_config.json` — constants, config, and metadata
  - **Verified artifact reload**: all artifacts loaded from disk and the complete inference pipeline reproduced known predictions with **10/10 matches** on sample data and **identical predictions/probabilities** across the full 3,000-sample test set (max probability difference < 1e-6).
  - Ran **13 artifact integrity checks** — 12/13 PASS, 1 N/A (MLflow not installed).
  - Established **MLflow traceability** (optional) with experiment logging for parameters, metrics, and artifacts.
  - Generated a **pinned `requirements.txt`** with exact versions for reproducibility.
  - Documented the **Day 2 handoff** — artifact loading instructions and the `/predict` endpoint contract for the FastAPI service.
* **Tools used:** `scikit-learn` (`TfidfVectorizer`, `LogisticRegression`, `accuracy_score`, `f1_score`), `joblib` (serialization), NLTK (tokenization, stopwords), `qalsadi` (Arabic lemmatizer), NumPy, Pandas, JSON, MLflow (optional).

### ✅ Day 2: Serving the Model with FastAPI

* **Objective:** Building a local REST API with FastAPI to serve the serialized Arabic sentiment classifier, with Pydantic validation, preprocessing reuse, and full notebook-vs-API consistency verification.
* **Key Tasks & Accomplishments:**
  - Built a **FastAPI application** (`main.py`) with three endpoints: `GET /` (health check), `GET /health` (lightweight probe), and `POST /predict` (sentiment classification).
  - Defined **Pydantic request/response schemas** — `PredictionRequest` with empty-text validation, `PredictionResponse` with prediction, label, confidence, probabilities, model version, and preprocessing step count.
  - Created a **shared preprocessing module** (`preprocessing.py`) with reusable `preprocess_text()` and `load_artifacts()` functions, imported by both the FastAPI app and the Day 3 Streamlit dashboard.
  - Loaded all **Day 1 artifacts** (`model.joblib`, `vectorizer.joblib`, `lemma_table.json`, `preprocessing_config.json`) at server startup, cached in a module-level variable.
  - Implemented the **full inference pipeline**: raw text → `preprocess_text()` → `vectorizer.transform()` → `model.predict()` → JSON response.
  - Verified **notebook-vs-API consistency** — 6 Arabic test samples produced identical predictions and probabilities (10/10 match).
  - Documented API usage in `FastAPI.ipynb` with Swagger UI at `/docs`.
* **Tools used:** FastAPI, Pydantic, Uvicorn, `scikit-learn` (`TfidfVectorizer`, `LogisticRegression`), `joblib`, NLTK, `qalsadi`, NumPy, JSON.

### ✅ Day 3: Interactive Streamlit Dashboard

* **Objective:** Building an interactive Streamlit dashboard that serves the trained Arabic sentiment classifier to non-technical users, with a clean demo UI suitable for live presentation.
* **Key Tasks & Accomplishments:**
  - Built a **Streamlit dashboard** (`app.py`) with text area input, example selector dropdown, prediction button, and result display using `st.success`/`st.error` with confidence percentages.
  - Implemented a **Matplotlib horizontal bar chart** for class probability visualization.
  - Reused the **shared preprocessing module** from Day 2 (`preprocessing.py`) — exact same pipeline as training.
  - Applied **`@st.cache_resource`** for artifact loading (load once, not on every rerun).
  - Validated **prediction consistency** — 6 Arabic test samples matched between notebook and Streamlit app (6/6 match).
  - Ran **8 functional test cases** covering positive/negative sentiment across different domains.
  - Tested **error handling** for empty string, whitespace, Latin text, numbers-only, and None input.
  - Achieved **10/10 validation checks PASS** — artifacts load correctly, model is correct, inference works, no retraining occurred, predictions are deterministic.
  - Documented the complete Day 3 lab in `Dashboard.ipynb` with functional tests, consistency validation, and final validation summary.
* **Tools used:** Streamlit, Matplotlib, `scikit-learn` (`TfidfVectorizer`, `LogisticRegression`), `joblib`, NLTK, `qalsadi`, NumPy, JSON.
### Day 4: Public Deployment *(Planned)*
### Day 5: Final Verification & Sprint Review *(Planned)*

---

### Final Production Model

| Property | Value |
|:---------|:------|
| **Model** | Logistic Regression (C=1.0, max_iter=1000, random_state=42) |
| **Representation** | TF-IDF (10K features, min_df=2, sublinear_tf=True) |
| **Test Accuracy** | 0.8623 |
| **Test Macro F1** | 0.8623 |
| **Test ROC-AUC** | 0.9417 |
| **5-fold CV F1** | 0.8556 ± 0.0033 |

### Serialized Artifacts

| Artifact | Size | Purpose |
|:---------|:-----|:--------|
| `artifacts/model.joblib` | ~79 KB | Trained Logistic Regression classifier |
| `artifacts/vectorizer.joblib` | ~386 KB | Fitted TF-IDF vectorizer (10K features) |
| `artifacts/lemma_table.json` | ~597 KB | Arabic token→lemma mapping (20K+ entries) |
| `artifacts/preprocessing_config.json` | ~2 KB | Constants, config, and metadata |
| **TOTAL** | **~1.04 MB** | |
