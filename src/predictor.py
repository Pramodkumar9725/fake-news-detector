"""
Inference & Explainability Engine for Fake News Detection.
Loads serialized pipeline, performs predictions with confidence calibration,
linguistic stylometric checks, and token-level attribution.
"""

import os
import json
import joblib
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.preprocessor import clean_text, extract_stylometric_features

class FakeNewsPredictor:
    def __init__(self, model_path=None, metrics_path=None):
        if model_path is None:
            model_path = os.path.join(PROJECT_ROOT, "models", "fake_news_pipeline.joblib")
        if metrics_path is None:
            metrics_path = os.path.join(PROJECT_ROOT, "models", "metrics.json")

        self.model_path = model_path
        self.metrics_path = metrics_path
        self.pipeline = None
        self.metrics = {}
        self.fake_tokens = {}
        self.real_tokens = {}
        self.load()

    def load(self):
        if os.path.exists(self.model_path):
            self.pipeline = joblib.load(self.model_path)
        else:
            self.pipeline = None

        if os.path.exists(self.metrics_path):
            with open(self.metrics_path, "r", encoding="utf-8") as f:
                self.metrics = json.load(f)
                explain = self.metrics.get("explainability", {})
                for item in explain.get("top_fake_tokens", []):
                    self.fake_tokens[item["token"]] = abs(item["weight"])
                for item in explain.get("top_real_tokens", []):
                    self.real_tokens[item["token"]] = abs(item["weight"])

    def is_ready(self):
        return self.pipeline is not None

    def predict(self, raw_text: str) -> dict:
        if not self.is_ready():
            self.load()
            if not self.is_ready():
                raise RuntimeError("Model pipeline has not been trained or loaded yet.")

        if not raw_text or not raw_text.strip():
            return {
                "error": "Empty text provided.",
                "prediction": "UNKNOWN",
                "confidence": 0.0
            }

        # 1. Linguistic & stylometric heuristic analysis
        stylometrics = extract_stylometric_features(raw_text)

        # 2. Text preprocessing
        cleaned = clean_text(raw_text, remove_stopwords=True)
        if not cleaned:
            cleaned = clean_text(raw_text, remove_stopwords=False)

        # 3. Model prediction
        probabilities = self.pipeline.predict_proba([cleaned])[0]
        # Class 0: FAKE, Class 1: REAL
        prob_fake = float(probabilities[0])
        prob_real = float(probabilities[1])

        # Sensationalism heuristic adjustment (mild prior calibration if high clickbait patterns exist)
        if stylometrics["sensational_score"] > 0.5:
            prob_fake = min(0.999, prob_fake + 0.05)
            prob_real = 1.0 - prob_fake

        if prob_real >= prob_fake:
            label = "REAL"
            confidence = prob_real
            verdict = "Credible / Reliable News"
            color_theme = "success"
        else:
            label = "FAKE"
            confidence = prob_fake
            verdict = "Deceptive / Fake News"
            color_theme = "danger"

        # Risk assessment rating
        if prob_fake >= 0.75:
            risk_level = "HIGH RISK"
        elif prob_fake >= 0.40:
            risk_level = "MODERATE RISK"
        else:
            risk_level = "LOW RISK"

        # 4. Explainability: identify influential tokens found in the input
        input_words = set(cleaned.split())
        matched_fake_cues = []
        matched_real_cues = []

        for word in input_words:
            if word in self.fake_tokens:
                matched_fake_cues.append({"word": word, "weight": self.fake_tokens[word], "type": "deceptive"})
            if word in self.real_tokens:
                matched_real_cues.append({"word": word, "weight": self.real_tokens[word], "type": "credible"})

        # Sort matches by weight
        matched_fake_cues.sort(key=lambda x: x["weight"], reverse=True)
        matched_real_cues.sort(key=lambda x: x["weight"], reverse=True)

        return {
            "prediction": label,
            "verdict": verdict,
            "confidence_percent": round(confidence * 100, 1),
            "probability_fake": round(prob_fake, 4),
            "probability_real": round(prob_real, 4),
            "risk_level": risk_level,
            "color_theme": color_theme,
            "stylometrics": stylometrics,
            "explanation": {
                "top_deceptive_cues": matched_fake_cues[:8],
                "top_credible_cues": matched_real_cues[:8],
                "sensational_keywords_found": stylometrics["sensational_terms_found"],
                "credibility_keywords_found": stylometrics["credibility_terms_found"]
            },
            "summary": f"Classified as {label} with {round(confidence * 100, 1)}% confidence ({risk_level})."
        }

    def predict_batch(self, texts: list) -> list:
        return [self.predict(t) for t in texts]
