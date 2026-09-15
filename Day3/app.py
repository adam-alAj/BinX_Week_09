"""
Arabic Sentiment Analysis — Interactive Streamlit Dashboard

Week 9 — Sprint 4 — Day 3

A clean, user-friendly dashboard that classifies Arabic review sentiment
using the trained TF-IDF + Logistic Regression model from Sprint 3/4.

Start with:  streamlit run app.py
"""

import sys
from pathlib import Path

# Add Day2 to path so preprocessing module is importable
sys.path.insert(0, str(Path(__file__).parent.parent / "Day2"))

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

from preprocessing import preprocess_text, load_artifacts

# ============================================================
# Page configuration
# ============================================================

st.set_page_config(
    page_title="Arabic Sentiment Analysis",
    page_icon="🇸🇦",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ============================================================
# Load model artifacts (cached)
# ============================================================


@st.cache_resource
def load_model_artifacts():
    """Load serialized model and preprocessing artifacts from Day 1."""
    return load_artifacts()


artifacts = load_model_artifacts()
model = artifacts["model"]
vectorizer = artifacts["vectorizer"]
lemma_table = artifacts["lemma_table"]
config = artifacts["config"]
label_names = config["label_names"]
label_mapping = {int(k): v for k, v in config["label_mapping"].items()}


# ============================================================
# Prediction function
# ============================================================


def predict_sentiment(raw_text):
    """
    Predict sentiment for raw Arabic text.
    Uses the exact same pipeline as training and Day 2.
    Returns: (prediction_int, label_str, probabilities_dict)
    """
    cleaned = preprocess_text(raw_text, lemma_table=lemma_table)
    features = vectorizer.transform([cleaned])
    prediction = int(model.predict(features)[0])
    proba = model.predict_proba(features)[0]
    probabilities = {label_names[i]: round(float(proba[i]), 4) for i in range(2)}
    return prediction, label_names[prediction], probabilities


# ============================================================
# UI: Header
# ============================================================

st.title("🇸🇦 Arabic Sentiment Analysis")
st.markdown("""
Enter an Arabic product review below, and the trained machine learning model
will predict whether the sentiment is **Positive** or **Negative**.
""")

st.divider()

# ============================================================
# UI: User Input
# ============================================================

# Example prompts
EXAMPLES = [
    "هذا المنتج ممتاز جدا وأنصح به للجميع",
    "المنتج سيء جدا ولا أنصح به أبداً",
    "جودة عالية وشحن سريع سعيد بالشراء",
]

st.subheader("Enter your review")

# Example selection
example_choice = st.selectbox(
    "Or try an example:",
    options=["(Custom input)"] + EXAMPLES,
    index=0,
    label_visibility="collapsed",
)

# Text area with default from example if selected
default_text = "" if example_choice == "(Custom input)" else example_choice

user_input = st.text_area(
    "Type or paste an Arabic review here:",
    value=default_text,
    height=120,
    placeholder="اكتب مراجعة باللغة العربية هنا...",
)

# ============================================================
# UI: Prediction Button & Result
# ============================================================

col_btn, col_spacer = st.columns([1, 3])
with col_btn:
    predict_clicked = st.button("🔍 Analyze Sentiment", type="primary", use_container_width=True)

if predict_clicked:
    if not user_input or not user_input.strip():
        st.warning("⚠️ Please enter some text before making a prediction.")
    else:
        try:
            prediction, label, probabilities = predict_sentiment(user_input)

            st.divider()
            st.subheader("Result")

            # Show prediction prominently
            if prediction == 1:
                st.success(f"**✅ Positive Sentiment** — {probabilities['Positive (1)']:.1%} confidence")
            else:
                st.error(f"**❌ Negative Sentiment** — {probabilities['Negative (0)']:.1%} confidence")

            # Metrics row
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Prediction", label)
            with col2:
                confidence = probabilities[label_names[prediction]]
                st.metric("Confidence", f"{confidence:.1%}")

            # ============================================================
            # Visualization: Probability Bar Chart
            # ============================================================

            st.subheader("Probability Distribution")

            neg_prob = probabilities["Negative (0)"]
            pos_prob = probabilities["Positive (1)"]

            fig, ax = plt.subplots(figsize=(8, 2.5))

            # Horizontal bar chart
            colors = ["#e74c3c", "#2ecc71"]
            bars = ax.barh(["Negative", "Positive"], [neg_prob, pos_prob], color=colors, height=0.6)

            # Add percentage labels on bars
            for bar, prob in zip(bars, [neg_prob, pos_prob]):
                width = bar.get_width()
                ax.text(
                    width + 0.01,
                    bar.get_y() + bar.get_height() / 2,
                    f"{prob:.1%}",
                    va="center",
                    ha="left",
                    fontsize=12,
                    fontweight="bold",
                )

            ax.set_xlim(0, 1.1)
            ax.set_xlabel("Probability")
            ax.set_title("Model Confidence by Class")
            ax.axvline(x=0.5, color="gray", linestyle="--", alpha=0.5, label="Decision boundary")
            ax.legend(loc="lower right")

            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

            # ============================================================
            # Explanation
            # ============================================================

            st.divider()
            st.subheader("How it works")

            st.markdown("""
            1. **Preprocessing** — Your text is normalized (diacritics removed, Alef unified,
               punctuation cleaned), tokenized, lemmatized, and stop words are removed.
            2. **TF-IDF Vectorization** — The cleaned text is converted to a 10,000-feature
               numerical representation using the fitted vectorizer.
            3. **Logistic Regression** — The model classifies the sentiment as Negative or Positive
               with associated probabilities.
            """)

            # Show model info
            with st.expander("Model Details"):
                st.markdown(f"""
                - **Model**: {config['model_config']['type']} (C={config['model_config']['C']})
                - **Features**: TF-IDF with {config['tfidf_config']['max_features']:,} features
                - **Training samples**: {config['dataset']['train_samples']:,}
                - **Test Accuracy**: {config['test_metrics']['accuracy']:.1%}
                - **Test F1 Score**: {config['test_metrics']['macro_f1']:.1%}
                - **Artifacts Version**: {config['artifacts_version']}
                """)

        except Exception as e:
            st.error(f"An error occurred during prediction: {str(e)}")

# ============================================================
# Footer
# ============================================================

st.divider()
st.caption(
    "Built with Streamlit | Model: TF-IDF + Logistic Regression | "
    "Sprint 4 — Week 9 | BinXTech AI & ML Internship"
)
