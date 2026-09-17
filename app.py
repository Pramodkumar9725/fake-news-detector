"""
Flask Web Application & REST API for Fake News Detection.
Serves interactive dashboard, live prediction API, batch CSV processor,
and model performance analytics.
"""

import os
import io
import csv
import json
from flask import Flask, request, jsonify, render_template, send_file, Response
import sys

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.predictor import FakeNewsPredictor

app = Flask(__name__, template_folder="templates", static_folder="static")
predictor = FakeNewsPredictor()

SAMPLE_ARTICLES = [
    {
        "type": "real",
        "category": "Science",
        "label": "Real News — James Webb Telescope",
        "title": "James Webb Space Telescope Observes Atmospheric Composition of Exoplanet",
        "text": "Astronomers utilizing spectroscopic data from the James Webb Space Telescope have identified carbon dioxide and sulfur dioxide in the atmosphere of a distant gas giant exoplanet. The findings, published in the journal Nature, provide new insights into atmospheric chemistry and planetary formation mechanisms occurring outside our solar system."
    },
    {
        "type": "real",
        "category": "Finance",
        "label": "Real News — Federal Reserve",
        "title": "Central Bank Holds Benchmark Interest Rates Steady Amid Cooling Inflation",
        "text": "The Federal Reserve announced that it would maintain the federal funds target rate in the 5.25 to 5.50 percent range following its two-day policy meeting. Chairman Jerome Powell stated that while inflation indicators have shown encouraging moderation, central bank governors will remain data-dependent before determining subsequent monetary easing timelines."
    },
    {
        "type": "fake",
        "category": "Health Hoax",
        "label": "Fake News — Miracle Cancer Cure",
        "title": "MIRACLE CURE: Secret Lemon Herb Completely Eradicates All Cancer in 24 Hours!",
        "text": "SHOCKING TRUTH BIG PHARMA DOES NOT WANT YOU TO KNOW! An ancient Himalayan herb combined with hot boiled lemon juice has been proven to permanently cure 100 percent of all terminal cancers within twenty-four hours! Evil corporate pharmaceutical executives have been murdering doctors who try to release this revolutionary miracle cure to save millions of human lives! Click here immediately before Google censors this video forever!"
    },
    {
        "type": "fake",
        "category": "Conspiracy",
        "label": "Fake News — 5G Mind Control",
        "title": "Whistleblower Exposes Secret 5G Towers Transmitting Mind Control Waves!",
        "text": "URGENT ALERT: A brave high-ranking government whistleblower has just leaked classified documents proving that cellular 5G transmitters are NOT communication towers, but clandestine psychological weapon arrays! The frequencies are engineered to alter human brainwaves and make civilians completely obedient to the New World Order puppet masters! Spread this viral report everywhere before they shut down our servers!"
    }
]

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/samples", methods=["GET"])
def get_samples():
    return jsonify({"status": "success", "samples": SAMPLE_ARTICLES})

@app.route("/api/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(silent=True) or {}
        text = data.get("text", "")
        if not text or not text.strip():
            return jsonify({"error": "Please provide non-empty text in the 'text' field."}), 400

        result = predictor.predict(text)
        return jsonify({"status": "success", "data": result})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/api/batch-predict", methods=["POST"])
def batch_predict():
    try:
        # Check if CSV file uploaded or JSON array provided
        if "file" in request.files:
            file = request.files["file"]
            if not file.filename.endswith(".csv"):
                return jsonify({"error": "Only CSV files are supported."}), 400

            stream = io.StringIO(file.stream.read().decode("UTF8"), newline=None)
            csv_reader = csv.DictReader(stream)

            # Detect text column
            fieldnames = csv_reader.fieldnames or []
            target_col = None
            for col in ["text", "content", "article", "news", "title", "headline"]:
                for fn in fieldnames:
                    if col in fn.lower():
                        target_col = fn
                        break
                if target_col:
                    break

            if not target_col and len(fieldnames) > 0:
                target_col = fieldnames[0]

            results = []
            for idx, row in enumerate(csv_reader):
                raw_text = row.get(target_col, "")
                if raw_text:
                    pred = predictor.predict(raw_text)
                    results.append({
                        "id": idx + 1,
                        "text": raw_text[:200] + ("..." if len(raw_text) > 200 else ""),
                        "prediction": pred["prediction"],
                        "confidence": pred["confidence_percent"],
                        "risk_level": pred["risk_level"],
                        "sensational_score": pred["stylometrics"]["sensational_score"]
                    })
                if len(results) >= 200: # Cap at 200 rows for preview
                    break

            return jsonify({
                "status": "success",
                "total_processed": len(results),
                "target_column_used": target_col,
                "results": results
            })

        data = request.get_json(silent=True) or {}
        texts = data.get("texts", [])
        if not isinstance(texts, list) or len(texts) == 0:
            return jsonify({"error": "Please upload a CSV file or provide an array of 'texts'."}), 400

        results = [predictor.predict(t) for t in texts[:100]]
        return jsonify({"status": "success", "total": len(results), "results": results})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/api/metrics", methods=["GET"])
def get_metrics():
    if not predictor.metrics:
        predictor.load()
    return jsonify(predictor.metrics)

@app.route("/api/sample-csv", methods=["GET"])
def download_sample_csv():
    """Generates a small downloadable sample CSV for batch testing."""
    sample_rows = [
        ["id", "headline_and_text"],
        ["1", "James Webb Space Telescope discovers water vapor and sulfur dioxide in distant exoplanet atmosphere, published by Nature."],
        ["2", "SHOCKING BOMBSHELL: Underground alien secret base discovered in Antarctica! Evil elites panic as whistleblowers leak footage!"],
        ["3", "Federal Reserve leaves benchmark interest rates unchanged at 5.50 percent following two-day monetary policy meeting."],
        ["4", "URGENT WARNING: Drinking hot lemon water mixed with Himalayan salt cures all cancer stages in 24 hours guaranteed!"],
        ["5", "Electoral Commission issues audited certified recount figures verifying voter turnout at 66.8 percent across districts."]
    ]
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerows(sample_rows)
    return Response(
        output.getvalue(),
        mimetype="text/csv",
        headers={"Content-disposition": "attachment; filename=sample_fake_news_batch.csv"}
    )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"\n==================================================")
    print(f"  Veritas AI — Fake News Detection Web Service")
    print(f"  Running locally on http://127.0.0.1:{port}")
    print(f"==================================================\n")
    app.run(host="127.0.0.1", port=port, debug=False)
