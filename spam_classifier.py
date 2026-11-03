"""
Project 1: Spam Email Detector
Classifies emails as HAM (legitimate) or SPAM using:
  1. Multinomial Naive Bayes
  2. Support Vector Machine (Linear & RBF SVM)

Features:
  - Text preprocessing & feature extraction via TF-IDF (1-gram and 2-gram)
  - 5-fold cross-validation and test set evaluation
  - Confusion Matrix, Precision, Recall, F1-Score, ROC-AUC
  - Top informative feature extraction (most spam-predictive vs. ham-predictive words)
  - Model serialization (joblib) and interactive prediction
"""

import os
import re
import tempfile

# Configure headless matplotlib environment before importing pyplot
os.environ["MPLCONFIGDIR"] = os.path.join(tempfile.gettempdir(), "matplotlib_cache")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC, SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    auc,
    precision_recall_curve,
    accuracy_score,
    f1_score,
)


def preprocess_text(text: str) -> str:
    """
    Cleans raw email text:
      - Lowercases
      - Normalizes URLs, emails, numbers, and currency symbols
      - Cleans punctuation and excess whitespace
    """
    if not isinstance(text, str):
        return ""
    text = text.lower()
    # Normalize URLs
    text = re.sub(r"https?://\S+|www\.\S+", " _url_ ", text)
    # Normalize email addresses
    text = re.sub(r"\b[\w\.-]+@[\w\.-]+\.\w+\b", " _email_ ", text)
    # Normalize dollar / currency amounts
    text = re.sub(r"[\$£€¥]\s*\d+([,\.]\d+)?", " _currency_ ", text)
    # Normalize numbers
    text = re.sub(r"\b\d+\b", " _number_ ", text)
    # Strip excess non-alphanumeric characters but keep tokens
    text = re.sub(r"[^a-zA-Z_\s]", " ", text)
    # Collapse multiple whitespaces
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_or_generate_data(data_path: str) -> pd.DataFrame:
    """Loads existing dataset or triggers generator if not found."""
    if not os.path.exists(data_path):
        print(f"Dataset not found at {data_path}. Generating fresh dataset...")
        from dataset.generate_spam_dataset import generate_dataset
        return generate_dataset(n_samples=800, output_path=data_path)
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} samples from {data_path}")
    return df


def train_and_evaluate(data_path: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    df = load_or_generate_data(data_path)

    # Clean text column
    df["clean_text"] = df["text"].apply(preprocess_text)
    X = df["clean_text"].values
    y = df["is_spam"].values  # 0 = Ham, 1 = Spam

    # Stratified Train/Test Split (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    print(f"\n--- Dataset Split ---")
    print(f"Training samples: {len(X_train)} ({np.sum(y_train == 0)} ham, {np.sum(y_train == 1)} spam)")
    print(f"Testing samples:  {len(X_test)} ({np.sum(y_test == 0)} ham, {np.sum(y_test == 1)} spam)")

    # Define TF-IDF vectorizer configuration
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=4000,
        sublinear_tf=True,
        stop_words="english",
    )

    # 1. Pipeline: Multinomial Naive Bayes
    nb_pipeline = Pipeline([
        ("tfidf", vectorizer),
        ("clf", MultinomialNB(alpha=0.1)),
    ])

    # 2. Pipeline: Linear Support Vector Machine
    svm_pipeline = Pipeline([
        ("tfidf", vectorizer),
        ("clf", LinearSVC(C=1.0, random_state=42, max_iter=2000)),
    ])

    # 3. Pipeline: Probability-calibrated Linear SVM (for ROC/confidence scores)
    svm_prob_pipeline = Pipeline([
        ("tfidf", vectorizer),
        ("clf", CalibratedClassifierCV(LinearSVC(C=1.0, random_state=42), ensemble=False)),
    ])

    # Perform 5-Fold Stratified Cross-Validation on Training Data
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    nb_cv_scores = cross_val_score(nb_pipeline, X_train, y_train, cv=cv, scoring="accuracy")
    svm_cv_scores = cross_val_score(svm_pipeline, X_train, y_train, cv=cv, scoring="accuracy")

    print("\n--- 5-Fold Cross-Validation Accuracy ---")
    print(f"Multinomial Naive Bayes: {nb_cv_scores.mean():.4f} (+/- {nb_cv_scores.std():.4f})")
    print(f"Support Vector Machine:  {svm_cv_scores.mean():.4f} (+/- {svm_cv_scores.std():.4f})")

    # Fit both models on full training set
    nb_pipeline.fit(X_train, y_train)
    svm_pipeline.fit(X_train, y_train)
    svm_prob_pipeline.fit(X_train, y_train)

    # Predictions
    y_pred_nb = nb_pipeline.predict(X_test)
    y_pred_svm = svm_pipeline.predict(X_test)

    # Probabilities for ROC
    y_prob_nb = nb_pipeline.predict_proba(X_test)[:, 1]
    y_prob_svm = svm_prob_pipeline.predict_proba(X_test)[:, 1]

    # Metrics evaluation
    print("\n=======================================================")
    print("        EVALUATION: MULTINOMIAL NAIVE BAYES            ")
    print("=======================================================")
    print(f"Test Accuracy: {accuracy_score(y_test, y_pred_nb):.4f}")
    print(f"F1 Score (Spam): {f1_score(y_test, y_pred_nb):.4f}")
    print("\nClassification Report:\n", classification_report(y_test, y_pred_nb, target_names=["Ham", "Spam"]))

    print("\n=======================================================")
    print("        EVALUATION: SUPPORT VECTOR MACHINE (SVM)        ")
    print("=======================================================")
    print(f"Test Accuracy: {accuracy_score(y_test, y_pred_svm):.4f}")
    print(f"F1 Score (Spam): {f1_score(y_test, y_pred_svm):.4f}")
    print("\nClassification Report:\n", classification_report(y_test, y_pred_svm, target_names=["Ham", "Spam"]))

    # Save best performing model
    model_save_path = os.path.join(output_dir, "spam_classifier_svm.joblib")
    joblib.dump(svm_prob_pipeline, model_save_path)
    print(f"\nSaved best model pipeline to: {model_save_path}")

    # Generate Evaluation Visualizations
    plot_evaluation_dashboard(
        X_test, y_test,
        y_pred_nb, y_pred_svm,
        y_prob_nb, y_prob_svm,
        svm_pipeline,
        output_dir,
    )

    return svm_prob_pipeline


def plot_evaluation_dashboard(X_test, y_test, y_pred_nb, y_pred_svm, y_prob_nb, y_prob_svm, svm_pipeline, output_dir):
    """Generates and saves a 4-panel comprehensive evaluation dashboard figure."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    fig.suptitle("Project 1: Spam Email Detection - Model Performance Dashboard", fontsize=15, fontweight="bold", y=0.98)

    # Panel 1: Confusion Matrix for Naive Bayes
    cm_nb = confusion_matrix(y_test, y_pred_nb)
    disp_nb = ConfusionMatrixDisplay(confusion_matrix=cm_nb, display_labels=["Ham", "Spam"])
    disp_nb.plot(ax=axes[0, 0], cmap="Blues", colorbar=False)
    axes[0, 0].set_title(f"Confusion Matrix: Naive Bayes\n(Accuracy: {accuracy_score(y_test, y_pred_nb)*100:.1f}%)", fontweight="bold")

    # Panel 2: Confusion Matrix for SVM
    cm_svm = confusion_matrix(y_test, y_pred_svm)
    disp_svm = ConfusionMatrixDisplay(confusion_matrix=cm_svm, display_labels=["Ham", "Spam"])
    disp_svm.plot(ax=axes[0, 1], cmap="Greens", colorbar=False)
    axes[0, 1].set_title(f"Confusion Matrix: Support Vector Machine\n(Accuracy: {accuracy_score(y_test, y_pred_svm)*100:.1f}%)", fontweight="bold")

    # Panel 3: ROC Curves
    fpr_nb, tpr_nb, _ = roc_curve(y_test, y_prob_nb)
    auc_nb = auc(fpr_nb, tpr_nb)
    fpr_svm, tpr_svm, _ = roc_curve(y_test, y_prob_svm)
    auc_svm = auc(fpr_svm, tpr_svm)

    axes[1, 0].plot(fpr_nb, tpr_nb, color="#1f77b4", lw=2, label=f"Naive Bayes (AUC = {auc_nb:.3f})")
    axes[1, 0].plot(fpr_svm, tpr_svm, color="#2ca02c", lw=2, label=f"SVM (AUC = {auc_svm:.3f})")
    axes[1, 0].plot([0, 1], [0, 1], color="gray", linestyle="--", lw=1)
    axes[1, 0].set_xlim([0.0, 1.0])
    axes[1, 0].set_ylim([0.0, 1.05])
    axes[1, 0].set_xlabel("False Positive Rate (FPR)", fontweight="bold")
    axes[1, 0].set_ylabel("True Positive Rate (TPR / Recall)", fontweight="bold")
    axes[1, 0].set_title("Receiver Operating Characteristic (ROC) Curve", fontweight="bold")
    axes[1, 0].legend(loc="lower right")
    axes[1, 0].grid(True, alpha=0.3)

    # Panel 4: Top Features from SVM
    tfidf = svm_pipeline.named_steps["tfidf"]
    clf = svm_pipeline.named_steps["clf"]
    feature_names = np.array(tfidf.get_feature_names_out())
    coefs = clf.coef_.ravel()

    # Top 8 spam indicators (positive coefficients) and top 8 ham indicators (negative coefficients)
    top_spam_indices = np.argsort(coefs)[-8:]
    top_ham_indices = np.argsort(coefs)[:8]

    top_features = np.concatenate([top_ham_indices, top_spam_indices])
    top_names = feature_names[top_features]
    top_weights = coefs[top_features]
    colors = ["#2b8cbe" if w < 0 else "#e34a33" for w in top_weights]

    y_pos = np.arange(len(top_names))
    axes[1, 1].barh(y_pos, top_weights, color=colors)
    axes[1, 1].set_yticks(y_pos)
    axes[1, 1].set_yticklabels(top_names, fontsize=9)
    axes[1, 1].axvline(0, color="black", lw=0.8, linestyle="--")
    axes[1, 1].set_xlabel("SVM Linear Coefficient (Weight)", fontweight="bold")
    axes[1, 1].set_title("Top 8 Ham (Blue) vs. Spam (Red) Keywords", fontweight="bold")
    axes[1, 1].grid(True, axis="x", alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, "spam_detector_metrics.png")
    plt.savefig(plot_path, dpi=200)
    plt.close()
    print(f"Generated evaluation dashboard plot: {plot_path}")


def predict_email(model, raw_email_text: str):
    """Predicts whether an email text is spam or ham with confidence score."""
    clean = preprocess_text(raw_email_text)
    prob = model.predict_proba([clean])[0]
    pred_label = "SPAM" if prob[1] >= 0.5 else "HAM"
    confidence = prob[1] if pred_label == "SPAM" else prob[0]
    return pred_label, confidence, prob[1]


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_csv = os.path.join(current_dir, "dataset", "spam_ham_dataset.csv")

    trained_model = train_and_evaluate(dataset_csv, output_dir=current_dir)

    print("\n--- Testing on Unseen Sample Emails ---")
    test_emails = [
        "Subject: Urgent: Security breach detected in your crypto wallet! Click here to verify 12-word seed phrase.",
        "Subject: Team Sync on Friday: We need to finalize the quarterly release roadmap and review PR #442.",
        "Subject: CONGRATULATIONS! You won a $1,000 Walmart gift card! Click here now to claim your prize!",
        "Subject: Attached invoice #4819 for cloud compute usage. Payment due within 30 days.",
    ]

    for email in test_emails:
        lbl, conf, spam_p = predict_email(trained_model, email)
        print(f"\nEmail: {email}")
        print(f"--> Prediction: [{lbl}] (Confidence: {conf*100:.1f}%, Spam Probability: {spam_p:.3f})")
