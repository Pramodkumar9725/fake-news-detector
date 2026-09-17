"""
Text Preprocessor & Feature Extraction Module for Fake News Detection.
Includes text cleaning, normalization, stylometric heuristics, and clickbait analysis.
Compatible with Scikit-Learn Pipelines.
"""

import re
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

# Comprehensive built-in English stopwords (no external NLTK downloads required)
STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
    "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being",
    "below", "between", "both", "but", "by", "can't", "cannot", "could", "couldn't",
    "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down", "during",
    "each", "few", "for", "from", "further", "had", "hadn't", "has", "hasn't",
    "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here",
    "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i",
    "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it",
    "it's", "its", "itself", "let's", "me", "more", "most", "mustn't", "my",
    "myself", "no", "nor", "not", "of", "off", "on", "once", "only", "or",
    "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same",
    "shan't", "she", "she'd", "she'll", "she's", "should", "shouldn't", "so",
    "some", "such", "than", "that", "that's", "the", "their", "theirs", "them",
    "themselves", "then", "there", "there's", "these", "they", "they'd", "they'll",
    "they're", "they've", "this", "those", "through", "to", "too", "under", "until",
    "up", "very", "was", "wasn't", "we", "we'd", "we'll", "we're", "we've", "were",
    "weren't", "what", "what's", "when", "when's", "where", "where's", "which", "while",
    "who", "who's", "whom", "why", "why's", "with", "won't", "would", "wouldn't", "you",
    "you'd", "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves"
}

# Common English contractions dictionary
CONTRACTIONS = {
    r"\bcan\'t\b": "cannot",
    r"\bwon\'t\b": "will not",
    r"\bn\'t\b": " not",
    r"\b\'re\b": " are",
    r"\b\'s\b": " is",
    r"\b\'d\b": " would",
    r"\b\'ll\b": " will",
    r"\b\'t\b": " not",
    r"\b\'ve\b": " have",
    r"\b\'m\b": " am"
}

# Sensationalist & clickbait keywords often correlated with deceptive content
SENSATIONAL_KEYWORDS = [
    "shocking", "miracle", "bombshell", "unbelievable", "secret", "banned",
    "censored", "leaked", "conspiracy", "sheeple", "reptilian", "nanobots",
    "deep state", "mind control", "alien", "pyramid", "elites", "cabal",
    "wake up", "urgent alert", "exposed", "treason", "tribunal", "chemtrail",
    "free energy", "eradicates", "hoax", "hidden truth", "must watch", "panic"
]

# Credibility indicators commonly found in verified journalistic coverage
CREDIBILITY_KEYWORDS = [
    "spokesperson", "reuters", "associated press", "announced", "confirmed",
    "official", "statement", "briefing", "investigation", "department",
    "committee", "testimony", "regulators", "published", "journal", "peer-reviewed",
    "audited", "recount", "electoral", "prosecutors", "statutory", "unanimous"
]

def clean_text(text: str, remove_stopwords: bool = True) -> str:
    """
    Cleans and normalizes raw text for vectorization and machine learning models.
    """
    if not isinstance(text, str):
        return ""

    # Remove HTML tags
    text = re.sub(r"<[^>]+>", " ", text)

    # Remove URLs
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)

    # Remove emails
    text = re.sub(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", " ", text)

    # Expand common contractions
    for pattern, replacement in CONTRACTIONS.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

    # Lowercase
    text = text.lower()

    # Remove non-alphabetic characters (retain whitespace)
    text = re.sub(r"[^a-z\s]", " ", text)

    # Remove excess whitespace
    tokens = text.split()

    # Stopwords removal
    if remove_stopwords:
        tokens = [token for token in tokens if token not in STOPWORDS and len(token) > 1]

    return " ".join(tokens)

def extract_stylometric_features(raw_text: str) -> dict:
    """
    Extracts stylometric and linguistic cues from raw uncleaned text:
    - Uppercase ratio (all caps / shouting)
    - Exclamation mark count
    - Question mark count
    - Sensationalism score
    - Credibility score
    - Character and word counts
    """
    if not isinstance(raw_text, str) or len(raw_text.strip()) == 0:
        return {
            "char_count": 0,
            "word_count": 0,
            "uppercase_ratio": 0.0,
            "exclamation_count": 0,
            "question_count": 0,
            "sensational_score": 0.0,
            "credibility_score": 0.0,
            "sensational_terms_found": [],
            "credibility_terms_found": []
        }

    char_count = len(raw_text)
    words = raw_text.split()
    word_count = len(words)

    # Uppercase ratio
    alpha_chars = [c for c in raw_text if c.isalpha()]
    upper_chars = [c for c in alpha_chars if c.isupper()]
    uppercase_ratio = (len(upper_chars) / len(alpha_chars)) if alpha_chars else 0.0

    # Punctuation counts
    exclamation_count = raw_text.count("!")
    question_count = raw_text.count("?")

    # Keyword searches (case-insensitive)
    lower_text = raw_text.lower()
    sensational_found = [term for term in SENSATIONAL_KEYWORDS if term in lower_text]
    credibility_found = [term for term in CREDIBILITY_KEYWORDS if term in lower_text]

    sensational_score = min(1.0, len(sensational_found) / 4.0)
    credibility_score = min(1.0, len(credibility_found) / 4.0)

    return {
        "char_count": char_count,
        "word_count": word_count,
        "uppercase_ratio": round(uppercase_ratio, 3),
        "exclamation_count": exclamation_count,
        "question_count": question_count,
        "sensational_score": round(sensational_score, 2),
        "credibility_score": round(credibility_score, 2),
        "sensational_terms_found": sensational_found,
        "credibility_terms_found": credibility_found
    }

class TextCleanerTransformer(BaseEstimator, TransformerMixin):
    """
    Scikit-learn compatible transformer for text cleaning.
    """
    def __init__(self, remove_stopwords: bool = True):
        self.remove_stopwords = remove_stopwords

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return [clean_text(str(text), self.remove_stopwords) for text in X]
