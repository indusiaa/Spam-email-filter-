"""
Test Samples & Interactive Email Spam Predictor
Load the serialized model pipeline and predict on custom or pre-set emails.
"""

import os
import sys
import tempfile
os.environ["MPLCONFIGDIR"] = os.path.join(tempfile.gettempdir(), "matplotlib_cache")

import joblib
from spam_classifier import preprocess_text, predict_email

def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, "spam_classifier_svm.joblib")

    if not os.path.exists(model_path):
        print(f"Model file not found at {model_path}. Please run spam_classifier.py first.")
        return

    model = joblib.load(model_path)
    print("=" * 65)
    print("      SPAM EMAIL CLASSIFIER - INTERACTIVE PREDICTION TOOL      ")
    print("=" * 65)

    sample_tests = [
        ("Meeting Notes", "Hi team,\nHere are the action items from today's daily standup. Please review the PRs before end of day.\nBest,\nSarah"),
        ("Crypto Airdrop", "CLAIM YOUR 5,000 FREE USDT NOW! Guaranteed passive return. Click http://airdrop-claim-now.xyz to claim!"),
        ("Invoice Due", "Dear Client,\nPlease find attached invoice #9021 for web development services. Payment is due in 15 days.\nRegards,\nFinance"),
        ("Lottery Alert", "URGENT WINNER: You have won $10,000,000 in the UK National Sweepstakes! Reply with your bank info immediately."),
        ("Password Reset", "Security Notice: Your password was recently changed. If this wasn't you, reset it via your security settings dashboard."),
    ]

    print("\n[Evaluating Benchmark Test Samples]")
    for title, email_body in sample_tests:
        label, conf, spam_p = predict_email(model, email_body)
        status_symbol = "[SPAM]" if label == "SPAM" else "[HAM]"
        print(f"\nScenario: {title}")
        print(f"Content snippet: {email_body[:80]}...")
        print(f"Result: {status_symbol} (Confidence: {conf*100:.1f}%, Spam Probability: {spam_p:.3f})")

    if len(sys.argv) > 1:
        custom_input = " ".join(sys.argv[1:])
        print(f"\n[Evaluating Custom CLI Argument]")
        label, conf, spam_p = predict_email(model, custom_input)
        print(f"Input: {custom_input}")
        print(f"Result: [{label}] (Confidence: {conf*100:.1f}%, Spam Prob: {spam_p:.3f})")

if __name__ == "__main__":
    main()
