# 🛡️ Veritas AI — Fake News Detection System using Machine Learning

An end-to-end Natural Language Processing (NLP) and Machine Learning application that detects fake, deceptive, or misleading news articles and headlines with high accuracy, calibrated confidence scores, stylometric risk heuristics, and word-level explainability.

---

## 🌟 Key Features

1. **Multiple Machine Learning Algorithms Benchmarked**:
   - **Passive-Aggressive Classifier (PAC)**: Margin-based linear model optimized for sparse text classification streams, calibrated for probabilistic outputs via `CalibratedClassifierCV`.
   - **Multinomial Naive Bayes (MNB)**: Fast probabilistic NLP baseline.
   - **Logistic Regression (LR)**: Linear model with L2 regularization and explainable feature coefficients.
   - **Voting Ensemble**: Hybrid ensemble combining probability distributions across candidate models.

2. **Advanced NLP & Feature Engineering**:
   - Automated text cleaning, punctuation handling, and contraction expansion.
   - Unigram and Bigram TF-IDF vectorization (`ngram_range=(1, 2)` with 5,000 sublinear TF features).
   - Built-in zero-dependency English stopword filtration.
   - **Stylometric & Heuristic Extractor**: Quantifies all-caps shouting ratio, exclamation/question punctuation frequency, sensationalist clickbait cues, and credibility indicators.

3. **Explainability & Attribution**:
   - Pinpoints exactly which words in the user's input contributed positively or negatively toward the classification decision with calculated weights.
   - Detects sensationalist keywords (e.g. *shocking, bombshell, miracle, suppressed*) and journalistic markers (e.g. *confirmed, spokesperson, published, peer-reviewed*).

4. **Dual Interfaces**:
   - **Modern Web Dashboard**: Glassmorphic UI with dark/light aesthetics, live single-article detector, batch CSV file processor with table filtering & CSV export, and an interactive model performance tab.
   - **Command-Line Interface (CLI)**: Fast terminal tool for single-line predictions, file processing, and metric inspections.
   - **RESTful API**: JSON endpoints (`/api/predict`, `/api/batch-predict`, `/api/metrics`, `/api/samples`, `/api/sample-csv`).

---

## 📁 Project Structure

```
fake_news_detector/
├── app.py                      # Flask web application & REST API
├── cli.py                      # Command-line interface tool
├── requirements.txt            # Pinned dependencies
├── README.md                   # Complete documentation
├── data/
│   ├── generate_dataset.py     # Benchmark dataset curation & generator
│   └── news_dataset.csv        # 700-sample balanced dataset (Real & Fake)
├── src/
│   ├── preprocessor.py         # Text cleaner, contractions, stylometrics, sklearn transformer
│   ├── train.py                # Model training, CV evaluation, calibration, serialization
│   └── predictor.py            # Inference engine, explainability, batch predictions
├── models/
│   ├── fake_news_pipeline.joblib # Serialized scikit-learn best pipeline
│   └── metrics.json            # Model benchmark evaluation metrics & explainability weights
├── templates/
│   └── index.html              # Modern responsive web dashboard
├── static/
│   ├── style.css               # Modern glassmorphic styles
│   └── app.js                  # Frontend client application logic
└── tests/
    └── test_detector.py        # Automated test suite (12 tests)
```

---

## 🚀 Quick Start

### 1. Requirements & Setup
Ensure Python 3.9+ is installed. Install the dependencies:
```bash
pip install -r requirements.txt
```

### 2. Generate Dataset & Train Models
```bash
# Curate dataset
python data/generate_dataset.py

# Train & benchmark all candidate models
python src/train.py
```

### 3. Launch Web Application
```bash
python app.py
```
Open your browser and navigate to:
```
http://127.0.0.1:5000
```

### 4. Run via CLI
```bash
# Analyze text
python cli.py --text "Scientists publish peer-reviewed study confirming renewable energy efficiency."

# Analyze sensational claim
python cli.py --text "SHOCKING BOMBSHELL: Secret alien base discovered under Antarctic ice sheet!"

# Inspect model benchmarks
python cli.py --metrics
```

### 5. Run Automated Tests
```bash
python -m unittest discover -s tests -p "test_*.py" -v
```

---

## 🔌 REST API Reference

### `POST /api/predict`
Analyze a single news article or headline.

**Request:**
```json
{
  "text": "The Federal Reserve announced that it would maintain benchmark interest rates."
}
```

**Response:**
```json
{
  "status": "success",
  "data": {
    "prediction": "REAL",
    "verdict": "Credible / Reliable News",
    "confidence_percent": 95.2,
    "probability_real": 0.952,
    "probability_fake": 0.048,
    "risk_level": "LOW RISK",
    "stylometrics": {
      "word_count": 11,
      "uppercase_ratio": 0.02,
      "sensational_score": 0.0,
      "credibility_score": 0.75
    },
    "explanation": {
      "top_credible_cues": [{"word": "announced", "weight": 0.85}],
      "sensational_keywords_found": []
    }
  }
}
```

### `POST /api/batch-predict`
Analyze a CSV file by uploading `multipart/form-data` with key `file`, or send a JSON payload:
```json
{
  "texts": ["Article snippet 1...", "Article snippet 2..."]
}
```

### `GET /api/metrics`
Retrieve full cross-validation and test set metrics for all candidate algorithms.

---

## 📊 Evaluation Results (Test Set)

| Algorithm | CV F1-Score | Test Accuracy | Test Precision | Test Recall | Test F1 | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Passive-Aggressive (Calibrated)** | **1.0000** | **100.0%** | **100.0%** | **100.0%** | **1.0000** | **1.0000** |
| Multinomial Naive Bayes | 1.0000 | 100.0% | 100.0% | 100.0% | 1.0000 | 1.0000 |
| Logistic Regression | 1.0000 | 100.0% | 100.0% | 100.0% | 1.0000 | 1.0000 |
| Voting Ensemble | 1.0000 | 100.0% | 100.0% | 100.0% | 1.0000 | 1.0000 |
