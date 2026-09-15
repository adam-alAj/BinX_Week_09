"""
Shared preprocessing module for Arabic Sentiment Classification.

This module contains the exact preprocessing functions used during training
(Day 1 / Week 8) to ensure training/serving consistency.

Artifacts are loaded separately — this module provides only the
deterministic code-based preprocessing steps.
"""

import re
import json
from pathlib import Path

import nltk
nltk.download("punkt_tab", quiet=True)
nltk.download("stopwords", quiet=True)
from nltk.tokenize import word_tokenize

try:
    from qalsadi.lemmatizer import Lemmatizer
    HAS_QALSADI = True
except ImportError:
    HAS_QALSADI = False

# ============================================================
# Constants (exact copy from Week 8 Day 1 / Week 9 Day 1)
# ============================================================

DIGIT_RE = re.compile(r"^\d+$")
LATIN_RE = re.compile(r"^[a-zA-Z]+$")

NEGATION_WORDS = {
    "\u0644\u0627",   # لا
    "\u0644\u0645",   # لم
    "\u0644\u0646",   # لن
    "\u0644\u064a\u0633", # ليس
    "\u0645\u0627",   # ما
    "\u063a\u064a\u0631", # غير
    "\u0628\u0644\u0627", # بلا
    "\u062f\u0648\u0646", # دون
    "\u062d\u0627\u0634\u0627", # حاشا
}

INTENSIFIER_WORDS = {
    "\u062c\u062f\u0627\u064b",  # جداً
    "\u062c\u062f\u0627",         # جدا
    "\u0643\u062b\u064a\u0631\u0627\u064b", # كثيراً
    "\u0643\u062b\u064a\u0631\u0627",        # كثيرا
}

PROTECTED = NEGATION_WORDS | INTENSIFIER_WORDS
STOP_WORDS = set(nltk.corpus.stopwords.words("arabic"))

# ============================================================
# Preprocessing functions (exact copy from Week 8 / Week 9 Day 1)
# ============================================================


def normalize_text(text):
    """
    Normalize Arabic text: remove tashkeel, unify Alef variants,
    replace ta-marbuta/alef-maksura, remove tatweel, remove punctuation.
    """
    if not isinstance(text, str):
        return ""
    # Remove tashkeel (diacritics)
    text = re.sub(r"[\u0617-\u061a\u064b-\u0652]", "", text)
    # Unify Alef variants
    text = re.sub(r"[\u0622\u0623\u0625]", "\u0627", text)
    # Ta-marbuta -> ha, Alef-maksura -> Ya
    text = text.replace("\u0629", "\u0647").replace("\u0649", "\u064a")
    # Remove tatweel
    text = re.sub(r"\u0640", "", text)
    # Remove punctuation
    text = re.sub(r"[!?.,:;()\[\]{}\"\'\\/-]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def unify_alef(token):
    """Unify Alef variants and replace ta-marbuta/alef-maksura in a token."""
    token = re.sub(r"[\u0622\u0623\u0625]", "\u0627", token)
    return token.replace("\u0629", "\u0647").replace("\u0649", "\u064a")


def preprocess_text(raw_text, lemma_table=None):
    """
    Full text preprocessing pipeline:
      normalize -> tokenize -> filter digits/Latin -> unify_alef ->
      protect negations -> lemmatize -> remove stopwords
    """
    if not isinstance(raw_text, str):
        return ""
    if lemma_table is None:
        lemma_table = {}
    tokens = word_tokenize(normalize_text(raw_text))
    out = []
    for tok in tokens:
        if DIGIT_RE.match(tok):
            continue
        if LATIN_RE.match(tok):
            out.append(tok.lower())
            continue
        u = unify_alef(tok)
        if u in PROTECTED:
            out.append(u)
            continue
        l = unify_alef(lemma_table.get(tok, tok))
        if u in STOP_WORDS or l in STOP_WORDS:
            continue
        out.append(l)
    return " ".join(out)


# ============================================================
# Artifact loading helpers
# ============================================================


def get_artifacts_dir():
    """Return the project-relative artifacts directory."""
    return Path(__file__).parent.parent / "Day1" / "artifacts"


def load_artifacts(artifacts_dir=None):
    """
    Load all serialized artifacts from Day 1.

    Returns
    -------
    dict with keys: model, vectorizer, lemma_table, config, label_names, label_mapping
    """
    import joblib

    if artifacts_dir is None:
        artifacts_dir = get_artifacts_dir()
    artifacts_dir = Path(artifacts_dir)

    # Load model
    model = joblib.load(artifacts_dir / "model.joblib")

    # Load vectorizer
    vectorizer = joblib.load(artifacts_dir / "vectorizer.joblib")

    # Load lemma table
    with open(artifacts_dir / "lemma_table.json", "r", encoding="utf-8") as f:
        lemma_table = json.load(f)

    # Load preprocessing config
    with open(artifacts_dir / "preprocessing_config.json", "r", encoding="utf-8") as f:
        config = json.load(f)

    return {
        "model": model,
        "vectorizer": vectorizer,
        "lemma_table": lemma_table,
        "config": config,
        "label_names": config["label_names"],
        "label_mapping": {int(k): v for k, v in config["label_mapping"].items()},
    }
