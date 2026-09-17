"""
Command-Line Interface (CLI) for Fake News Detection.
Usage:
    python cli.py --text "Scientists publish new peer-reviewed findings on solar energy."
    python cli.py --file path/to/article.txt
    python cli.py --metrics
"""

import argparse
import sys
import os
import json

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.predictor import FakeNewsPredictor

def main():
    parser = argparse.ArgumentParser(description="Fake News Detection using Machine Learning")
    parser.add_argument("--text", type=str, help="News article or headline text to analyze")
    parser.add_argument("--file", type=str, help="Path to text file containing article")
    parser.add_argument("--metrics", action="store_true", help="Display model benchmark metrics")

    args = parser.parse_args()

    predictor = FakeNewsPredictor()

    if args.metrics:
        print("\n" + "=" * 60)
        print("MODEL BENCHMARK METRICS")
        print("=" * 60)
        if predictor.metrics:
            print(f"Best Model: {predictor.metrics.get('best_model')}")
            print(f"Dataset Size: {predictor.metrics.get('dataset_stats', {}).get('total_samples')} samples")
            print("\nCandidate Performance:")
            for model_name, m in predictor.metrics.get("models_benchmark", {}).items():
                print(f"  • {model_name:30s} | Acc: {m['test_accuracy']*100:.1f}% | F1: {m['test_f1']:.4f} | ROC-AUC: {m['test_roc_auc']:.4f}")
        else:
            print("No metrics file found. Run 'python src/train.py' first.")
        print("=" * 60 + "\n")
        return

    text_to_analyze = None
    if args.text:
        text_to_analyze = args.text
    elif args.file:
        if not os.path.exists(args.file):
            print(f"Error: File '{args.file}' not found.")
            sys.exit(1)
        with open(args.file, "r", encoding="utf-8") as f:
            text_to_analyze = f.read()

    if not text_to_analyze:
        print("Error: Please provide --text, --file, or --metrics. Use --help for options.")
        sys.exit(1)

    result = predictor.predict(text_to_analyze)

    print("\n" + "=" * 60)
    print("FAKE NEWS DETECTION REPORT")
    print("=" * 60)
    print(f"Verdict:             {result['verdict'].upper()}")
    print(f"Classification:      {result['prediction']}")
    print(f"Confidence:          {result['confidence_percent']}%")
    print(f"Risk Rating:         {result['risk_level']}")
    print(f"Probability (Real):  {result['probability_real'] * 100:.1f}%")
    print(f"Probability (Fake):  {result['probability_fake'] * 100:.1f}%")

    sty = result["stylometrics"]
    print("-" * 60)
    print("LINGUISTIC & STYLOMETRIC CUES:")
    print(f"  • Words: {sty['word_count']} | All-Caps Ratio: {sty['uppercase_ratio'] * 100:.1f}%")
    print(f"  • Exclamation Marks: {sty['exclamation_count']} | Question Marks: {sty['question_count']}")
    print(f"  • Sensationalism Score: {sty['sensational_score'] * 100:.0f}%")
    print(f"  • Credibility Score:    {sty['credibility_score'] * 100:.0f}%")

    exp = result["explanation"]
    if exp["sensational_keywords_found"]:
        print(f"  • Sensational Terms Detected: {', '.join(exp['sensational_keywords_found'])}")
    if exp["credibility_keywords_found"]:
        print(f"  • Credible Markers Detected:  {', '.join(exp['credibility_keywords_found'])}")

    if exp["top_deceptive_cues"]:
        print("\nTop Deceptive Influencers:")
        for cue in exp["top_deceptive_cues"][:5]:
            print(f"  [!] '{cue['word']}' (weight: {cue['weight']:.2f})")

    if exp["top_credible_cues"]:
        print("\nTop Credibility Influencers:")
        for cue in exp["top_credible_cues"][:5]:
            print(f"  [+] '{cue['word']}' (weight: {cue['weight']:.2f})")

    print("=" * 60 + "\n")

if __name__ == "__main__":
    main()
