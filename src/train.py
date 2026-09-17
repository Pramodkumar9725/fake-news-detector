"""
Model Training & Evaluation Pipeline for Fake News Detection.
Trains, evaluates, and compares multiple Machine Learning models:
- Passive-Aggressive Classifier (Calibrated for probability)
- Multinomial Naive Bayes
- Logistic Regression
- Voting Ensemble (Soft probability voting)

Evaluates accuracy, precision, recall, F1, ROC-AUC, and confusion matrix.
Serializes the best pipeline and benchmark metrics.
"""

import os
import json
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier, LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import VotingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

import sys
# Add project root to sys.path to allow imports
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.preprocessor import clean_text

def train_and_evaluate(
    data_path=os.path.join(PROJECT_ROOT, "data", "news_dataset.csv"),
    models_dir=os.path.join(PROJECT_ROOT, "models")
):
    os.makedirs(models_dir, exist_ok=True)

    print("=" * 60)
    print("1. LOADING AND PREPARING DATASET")
    print("=" * 60)
    if not os.path.exists(data_path):
        from data.generate_dataset import generate_dataset
        print(f"Dataset not found at {data_path}. Generating now...")
        generate_dataset(data_path)

    df = pd.read_csv(data_path)
    print(f"Total dataset records: {len(df)}")
    print(f"Class counts:\n{df['label_name'].value_counts()}")

    # Preprocess text corpus
    print("\nCleaning text samples...")
    df["cleaned_text"] = df["full_text"].apply(lambda t: clean_text(str(t), remove_stopwords=True))

    X = df["cleaned_text"]
    y = df["label"].values # 1: REAL, 0: FAKE

    # Stratified Train/Test Split (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"Train size: {len(X_train)} | Test size: {len(X_test)}")

    # Vectorizer
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=5000,
        sublinear_tf=True
    )

    print("\n" + "=" * 60)
    print("2. DEFINING CANDIDATE MACHINE LEARNING MODELS")
    print("=" * 60)

    # Base models
    pac_raw = PassiveAggressiveClassifier(max_iter=1000, C=0.5, random_state=42)
    pac_calibrated = CalibratedClassifierCV(pac_raw, cv=3)
    mnb = MultinomialNB(alpha=0.2)
    log_reg = LogisticRegression(C=1.0, max_iter=1000, random_state=42)

    # Soft Voting Ensemble (Combines probabilities from PAC, MNB, and LogReg)
    ensemble = VotingClassifier(
        estimators=[
            ('pac', pac_calibrated),
            ('mnb', mnb),
            ('lr', log_reg)
        ],
        voting='soft'
    )

    candidate_models = {
        "Passive-Aggressive": pac_calibrated,
        "Multinomial Naive Bayes": mnb,
        "Logistic Regression": log_reg,
        "Voting Ensemble (Best Hybrid)": ensemble
    }

    results = {}
    best_model_name = None
    best_f1 = -1.0
    best_pipeline = None

    print("\n" + "=" * 60)
    print("3. TRAINING, CROSS-VALIDATION & BENCHMARKING")
    print("=" * 60)

    for name, clf in candidate_models.items():
        pipeline = Pipeline([
            ("vectorizer", TfidfVectorizer(ngram_range=(1, 2), max_features=5000, sublinear_tf=True)),
            ("classifier", clf)
        ])

        # 5-fold cross validation on training set
        cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring="f1")
        
        # Fit on full training set
        pipeline.fit(X_train, y_train)
        
        # Predict on test set
        y_pred = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        roc_auc = roc_auc_score(y_test, y_proba)
        cm = confusion_matrix(y_test, y_pred).tolist()

        results[name] = {
            "cv_f1_mean": round(float(np.mean(cv_scores)), 4),
            "cv_f1_std": round(float(np.std(cv_scores)), 4),
            "test_accuracy": round(float(acc), 4),
            "test_precision": round(float(prec), 4),
            "test_recall": round(float(rec), 4),
            "test_f1": round(float(f1), 4),
            "test_roc_auc": round(float(roc_auc), 4),
            "confusion_matrix": cm
        }

        print(f"\n--- {name} ---")
        print(f"  CV F1 Score:  {results[name]['cv_f1_mean']:.4f} (+/- {results[name]['cv_f1_std']:.4f})")
        print(f"  Test Accuracy: {acc * 100:.2f}%")
        print(f"  Test F1:       {f1:.4f}")
        print(f"  Test ROC-AUC:  {roc_auc:.4f}")
        print(f"  Confusion Matrix (TN, FP / FN, TP): {cm}")

        if f1 > best_f1:
            best_f1 = f1
            best_model_name = name
            best_pipeline = pipeline

    print("\n" + "=" * 60)
    print(f"BEST MODEL SELECTED: {best_model_name} (Test F1: {best_f1:.4f})")
    print("=" * 60)

    # Save the best pipeline
    model_output_path = os.path.join(models_dir, "fake_news_pipeline.joblib")
    joblib.dump(best_pipeline, model_output_path)
    print(f"Saved best model pipeline to: {model_output_path}")

    # Also extract vocabulary & feature weights for explainability
    # Fit vectorizer and logistic regression standalone for direct token coefficients
    exp_vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=5000, sublinear_tf=True)
    X_train_vec = exp_vectorizer.fit_transform(X_train)
    exp_lr = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    exp_lr.fit(X_train_vec, y_train)

    feature_names = exp_vectorizer.get_feature_names_out()
    coefficients = exp_lr.coef_[0]

    top_fake_idx = np.argsort(coefficients)[:30]
    top_real_idx = np.argsort(coefficients)[::-1][:30]

    explainability_data = {
        "top_fake_tokens": [{"token": feature_names[i], "weight": round(float(coefficients[i]), 4)} for i in top_fake_idx],
        "top_real_tokens": [{"token": feature_names[i], "weight": round(float(coefficients[i]), 4)} for i in top_real_idx]
    }

    # Save metrics and explainability data
    metrics_data = {
        "best_model": best_model_name,
        "models_benchmark": results,
        "explainability": explainability_data,
        "dataset_stats": {
            "total_samples": int(len(df)),
            "train_samples": int(len(X_train)),
            "test_samples": int(len(X_test)),
            "real_samples": int((df['label'] == 1).sum()),
            "fake_samples": int((df['label'] == 0).sum())
        }
    }

    metrics_path = os.path.join(models_dir, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_data, f, indent=2)
    print(f"Saved metrics & explainability metadata to: {metrics_path}")

    return best_pipeline, metrics_data

if __name__ == "__main__":
    train_and_evaluate()
