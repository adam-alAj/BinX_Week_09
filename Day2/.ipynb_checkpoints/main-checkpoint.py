"""
FastAPI application for Arabic Sentiment Classification.

Loads serialized artifacts from Day 1 and serves predictions via REST API.

Start with:  uvicorn main:app --reload
Docs at:     http://127.0.0.1:8000/docs
"""

import os
import sys
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

# Add Day2 directory to path so preprocessing module is importable
sys.path.insert(0, str(Path(__file__).parent))

from preprocessing import preprocess_text, load_artifacts, get_artifacts_dir

# ============================================================
# Pydantic request/response schemas
# ============================================================


class PredictionRequest(BaseModel):
    """Request schema for /predict endpoint."""

    text: str = Field(
        ...,
        min_length=1,
        description="Arabic review text to classify",
        json_schema_extra={"example": "هذا المنتج ممتاز جدا"},
    )

    @field_validator("text")
    @classmethod
    def text_must_not_be_empty(cls, v):
        if not v.strip():
            raise ValueError("Text must not be empty or whitespace-only")
        return v


class PredictionResponse(BaseModel):
    """Response schema for /predict endpoint."""

    prediction: int = Field(description="Predicted class label (0 or 1)")
    label: str = Field(description="Human-readable sentiment label")
    confidence: float = Field(description="Confidence score for predicted class")
    probabilities: dict = Field(description="Probability for each class")
    model_version: str = Field(description="Artifacts version from config")
    preprocessing_steps: int = Field(
        description="Number of preprocessing steps applied"
    )


# ============================================================
# FastAPI application
# ============================================================

app = FastAPI(
    title="Arabic Sentiment Classification API",
    description="Serve Arabic sentiment predictions using TF-IDF + Logistic Regression.",
    version="1.0.0",
)

# ============================================================
# Load artifacts at startup (once)
# ============================================================

_artifacts = None


def _load():
    global _artifacts
    if _artifacts is None:
        _artifacts = load_artifacts()
    return _artifacts


# Load immediately so the app is ready when uvicorn starts
_load()


# ============================================================
# Endpoints
# ============================================================


@app.get("/")
def root():
    """Health / root endpoint confirming the API is running."""
    data = _load()
    return {
        "status": "ok",
        "message": "Arabic Sentiment Classification API",
        "model_loaded": True,
        "model_type": data["model"].__class__.__name__,
        "artifacts_version": data["config"]["artifacts_version"],
    }


@app.get("/health")
def health():
    """Lightweight health check."""
    return {"status": "healthy"}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    """
    Classify the sentiment of an Arabic review.

    Accepts raw Arabic text, applies the exact same preprocessing pipeline
    used during training, and returns the predicted sentiment with
    confidence scores.
    """
    data = _load()
    t0 = time.time()

    try:
        # Step 1: Preprocess (exact same function as training)
        cleaned = preprocess_text(request.text, lemma_table=data["lemma_table"])

        # Step 2: Vectorize (loaded fitted TF-IDF vectorizer)
        features = data["vectorizer"].transform([cleaned])

        # Step 3: Predict
        prediction = int(data["model"].predict(features)[0])

        # Step 4: Get probabilities
        proba = data["model"].predict_proba(features)[0]
        label = data["label_names"][prediction]
        probabilities = {
            data["label_names"][i]: round(float(proba[i]), 4) for i in range(2)
        }
        confidence = round(float(proba[prediction]), 4)

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}",
        )

    t_total = time.time() - t0

    return PredictionResponse(
        prediction=prediction,
        label=label,
        confidence=confidence,
        probabilities=probabilities,
        model_version=data["config"]["artifacts_version"],
        preprocessing_steps=len(data["config"]["preprocessing_steps"]),
    )


# ============================================================
# Entry point (for direct execution)
# ============================================================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
