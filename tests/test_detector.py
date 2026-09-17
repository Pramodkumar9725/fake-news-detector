"""
Automated Test Suite for Fake News Detection System.
Tests preprocessing, dataset integrity, machine learning inference, and Flask REST API.
"""

import os
import unittest
import json
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.preprocessor import clean_text, extract_stylometric_features, TextCleanerTransformer
from src.predictor import FakeNewsPredictor
from app import app

class TestTextPreprocessor(unittest.TestCase):
    def test_clean_text_basic(self):
        sample = "BREAKING: Check this out at https://fake.com! Don't miss <b>it</b>."
        cleaned = clean_text(sample, remove_stopwords=True)
        self.assertNotIn("https", cleaned)
        self.assertNotIn("<b>", cleaned)
        self.assertNotIn("don't", cleaned)
        self.assertIn("breaking", cleaned)

    def test_stylometric_features(self):
        text = "SHOCKING BOMBSHELL! Evil secret alien base discovered! Why hide it?"
        sty = extract_stylometric_features(text)
        self.assertGreater(sty["uppercase_ratio"], 0.15)
        self.assertGreaterEqual(sty["exclamation_count"], 2)
        self.assertEqual(sty["question_count"], 1)
        self.assertGreater(sty["sensational_score"], 0.4)
        self.assertIn("shocking", sty["sensational_terms_found"])

class TestModelPredictor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.predictor = FakeNewsPredictor()

    def test_predictor_ready(self):
        self.assertTrue(self.predictor.is_ready(), "Model pipeline should be loaded and ready.")

    def test_real_news_prediction(self):
        sample = (
            "The Department of Transportation and congressional leaders announced a bipartisan "
            "funding bill for municipal bridge repairs and public highway infrastructure."
        )
        res = self.predictor.predict(sample)
        self.assertEqual(res["prediction"], "REAL")
        self.assertGreaterEqual(res["confidence_percent"], 50.0)
        self.assertIn("LOW", res["risk_level"])
        self.assertIn("announced", res["explanation"]["credibility_keywords_found"])

    def test_fake_news_prediction(self):
        sample = (
            "SHOCKING BOMBSHELL: Whistleblowers leak secret footage proving evil alien overlords "
            "control the government and cancer can be cured with lemon juice in 24 hours!"
        )
        res = self.predictor.predict(sample)
        self.assertEqual(res["prediction"], "FAKE")
        self.assertGreaterEqual(res["confidence_percent"], 50.0)
        self.assertIn("HIGH", res["risk_level"])
        self.assertGreater(len(res["explanation"]["sensational_keywords_found"]), 0)

    def test_empty_input(self):
        res = self.predictor.predict("   ")
        self.assertIn("error", res)

class TestFlaskEndpoints(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = app.test_client()

    def test_index_route(self):
        res = self.client.get("/")
        self.assertEqual(res.status_code, 200)
        self.assertIn(b"Veritas", res.data)

    def test_api_samples(self):
        res = self.client.get("/api/samples")
        self.assertEqual(res.status_code, 200)
        data = json.loads(res.data)
        self.assertEqual(data["status"], "success")
        self.assertGreater(len(data["samples"]), 0)

    def test_api_predict_valid(self):
        payload = {"text": "Federal Reserve officials voted to maintain interest rates."}
        res = self.client.post("/api/predict", json=payload)
        self.assertEqual(res.status_code, 200)
        data = json.loads(res.data)
        self.assertEqual(data["status"], "success")
        self.assertIn("prediction", data["data"])

    def test_api_predict_empty(self):
        res = self.client.post("/api/predict", json={"text": ""})
        self.assertEqual(res.status_code, 400)

    def test_api_metrics(self):
        res = self.client.get("/api/metrics")
        self.assertEqual(res.status_code, 200)
        data = json.loads(res.data)
        self.assertIn("models_benchmark", data)

    def test_api_sample_csv(self):
        res = self.client.get("/api/sample-csv")
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.mimetype, "text/csv")
        self.assertIn(b"headline_and_text", res.data)

if __name__ == "__main__":
    unittest.main()
