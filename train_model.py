"""
train_model.py
--------------
End-to-end training script for the Spam Email Detection system.

Workflow
--------
1.  Load & inspect the dataset
2.  Clean, deduplicate, and validate the data
3.  Encode labels (ham=0, spam=1)
4.  Train / test split (stratified, 80/20)
5.  Fit TF-IDF vectoriser on TRAINING data only (no leakage)
6.  Train three classifiers:
        - Logistic Regression
        - Multinomial Naive Bayes
        - Calibrated Linear SVM  (CalibratedClassifierCV wrapping LinearSVC)
7.  Select best model by 5-fold cross-validated F1 on training data
    (avoids optimistic bias from selecting on the held-out test set)
8.  Re-fit best model on full training data; evaluate on held-out test set
9.  Save best model + vectoriser to models/
10. Save evaluation plots to outputs/

Run
---
    python train_model.py
"""

import os
import sys
import json
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # non-interactive backend — safe for headless runs
import matplotlib.pyplot as plt
import joblib

from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
)

# Add project root to sys.path so 'utils' package resolves correctly
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.preprocessing import clean_text, preprocess_dataframe

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────
# 0. Path configuration
# ──────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
DATA_PATH    = os.path.join(BASE_DIR, "data", "spam_ham_dataset.csv")
MODELS_DIR   = os.path.join(BASE_DIR, "models")
OUTPUTS_DIR  = os.path.join(BASE_DIR, "outputs")
MODEL_PATH   = os.path.join(MODELS_DIR, "spam_classifier.pkl")
VECTOR_PATH  = os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl")
METRICS_PATH = os.path.join(MODELS_DIR, "metrics.json")

os.makedirs(MODELS_DIR,  exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)

RANDOM_STATE = 42


# ──────────────────────────────────────────────
# 1. Load & inspect dataset
# ──────────────────────────────────────────────
def load_and_inspect(path: str) -> pd.DataFrame:
    print("\n" + "="*60)
    print("  STEP 1 - Loading & Inspecting Dataset")
    print("="*60)

    df = pd.read_csv(path)

    print(f"  Rows       : {df.shape[0]}")
    print(f"  Columns    : {df.shape[1]}  ->  {df.columns.tolist()}")
    print(f"  Dtypes     :\n{df.dtypes.to_string()}")
    print(f"\n  Missing values:\n{df.isnull().sum().to_string()}")
    print(f"\n  Duplicates : {df.duplicated().sum()}")
    print(f"\n  Label distribution:\n{df['label'].value_counts().to_string()}")
    print(f"\n  label_num distribution:\n{df['label_num'].value_counts().to_string()}")
    print("\n  Sample (first 2 rows):")
    print(df[["label", "label_num", "text"]].head(2).to_string())

    return df


# ──────────────────────────────────────────────
# 2. Clean & validate
# ──────────────────────────────────────────────
def prepare_data(df: pd.DataFrame):
    print("\n" + "="*60)
    print("  STEP 2 - Cleaning, Deduplicating & Preparing Data")
    print("="*60)

    raw_count = len(df)
    # The dataset already has 'text' and 'label_num' columns.
    # label_num:  0 = ham,  1 = spam
    # preprocess_dataframe now handles deduplication internally.
    clean_df = preprocess_dataframe(df, text_col="text", label_col="label_num")

    dropped = raw_count - len(clean_df)
    print(f"  Raw records            : {raw_count}")
    print(f"  Duplicate rows dropped : {dropped}")
    print(f"  Records after cleaning : {len(clean_df)}")
    print(f"  Ham  (0) : {(clean_df['label'] == 0).sum()}")
    print(f"  Spam (1) : {(clean_df['label'] == 1).sum()}")

    return clean_df


# ──────────────────────────────────────────────
# 3. Train / test split
# ──────────────────────────────────────────────
def split_data(df: pd.DataFrame):
    print("\n" + "="*60)
    print("  STEP 3 - Train / Test Split  (80 / 20, stratified)")
    print("="*60)

    X = df["text"]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y,     # preserves class ratio in both splits
    )

    print(f"  Training samples : {len(X_train)}")
    print(f"  Testing  samples : {len(X_test)}")
    print(f"  Train spam ratio : {y_train.mean():.3f}")
    print(f"  Test  spam ratio : {y_test.mean():.3f}")

    return X_train, X_test, y_train, y_test


# ──────────────────────────────────────────────
# 4. TF-IDF feature extraction
# ──────────────────────────────────────────────
def build_tfidf_features(X_train, X_test):
    """
    Why TF-IDF?
    -----------
    TF-IDF (Term Frequency–Inverse Document Frequency) converts raw text into
    numerical feature vectors while down-weighting very common words that
    appear in almost every document (and hence carry little discriminating
    information) and up-weighting rare but informative terms.

    Key parameters chosen:
    - ngram_range=(1,2)  : captures both single words AND two-word phrases
                           e.g. "click here", "free offer" are strong spam signals
    - min_df=2           : ignore tokens that appear in fewer than 2 documents
                           (likely typos / noise)
    - max_df=0.95        : ignore tokens that appear in >95% of documents
                           (too common to be discriminating)
    - sublinear_tf=True  : replaces raw TF with 1+log(TF) to dampen the effect
                           of extremely frequent terms within one document
    - stop_words="english": remove common English function words
    """
    print("\n" + "="*60)
    print("  STEP 4 - TF-IDF Feature Extraction")
    print("="*60)

    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
    )

    # CRITICAL: fit ONLY on training data to prevent data leakage
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf  = vectorizer.transform(X_test)

    print(f"  Vocabulary size      : {len(vectorizer.vocabulary_):,}")
    print(f"  Train feature matrix : {X_train_tfidf.shape}")
    print(f"  Test  feature matrix : {X_test_tfidf.shape}")

    return vectorizer, X_train_tfidf, X_test_tfidf


# ──────────────────────────────────────────────
# 5. Define models
# ──────────────────────────────────────────────
def get_models() -> dict:
    """
    Return a dict of {name: model_instance} for the three classifiers.

    - LogisticRegression  : strong linear baseline; natively calibrated proba.
    - MultinomialNB       : classic bag-of-words classifier; fast and effective.
    - Calibrated Linear SVM: LinearSVC wrapped in CalibratedClassifierCV so it
                             exposes a true predict_proba via Platt scaling.
                             This fixes the "confidence" bug where sigmoid(|decision|)
                             can never drop below 50% and is not calibrated.
    """
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            random_state=RANDOM_STATE,
            C=1.0,
            solver="lbfgs",
        ),
        "Naive Bayes": MultinomialNB(alpha=0.1),
        # CalibratedClassifierCV wraps LinearSVC so predict_proba is available.
        # cv=5 uses 5-fold internal cross-validation to fit the calibration layer.
        "Linear SVM": CalibratedClassifierCV(
            LinearSVC(max_iter=2000, random_state=RANDOM_STATE, C=1.0),
            cv=5,
        ),
    }


# ──────────────────────────────────────────────
# 6. Train & evaluate all models
# ──────────────────────────────────────────────
def evaluate_model(model, X_train_tfidf, X_test_tfidf, y_train, y_test) -> dict:
    model.fit(X_train_tfidf, y_train)
    y_pred = model.predict(X_test_tfidf)

    return {
        "accuracy" : round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall"   : round(recall_score(y_test, y_pred, zero_division=0), 4),
        "f1"       : round(f1_score(y_test, y_pred, zero_division=0), 4),
        "y_pred"   : y_pred,
    }


def train_and_compare(models, X_train_tfidf, X_test_tfidf, y_train, y_test):
    print("\n" + "="*60)
    print("  STEP 5 - Training & Evaluating All Models")
    print("="*60)

    results = {}
    for name, model in models.items():
        print(f"\n  Training: {name} ...")
        results[name] = evaluate_model(
            model, X_train_tfidf, X_test_tfidf, y_train, y_test
        )
        m = results[name]
        print(f"    Accuracy : {m['accuracy']:.4f}")
        print(f"    Precision: {m['precision']:.4f}")
        print(f"    Recall   : {m['recall']:.4f}")
        print(f"    F1 Score : {m['f1']:.4f}")
        print(f"\n  Classification Report ({name}):")
        print(classification_report(y_test, m["y_pred"],
                                    target_names=["HAM", "SPAM"]))

    return results


# ──────────────────────────────────────────────
# 7. Select best model via cross-validation
# ──────────────────────────────────────────────
def select_best_model(models: dict, results: dict,
                      X_train_tfidf, y_train):
    """
    Model selection uses 5-fold cross-validated F1 on the TRAINING data only.
    This avoids the optimistic bias of picking the winner based on the same
    held-out test set that we later report final scores on.

    The cv_f1 score is used only for selection; the test-set metrics in
    `results` are the honest final evaluation numbers.
    """
    print("\n" + "="*60)
    print("  STEP 6 - Cross-Validated Model Selection")
    print("="*60)

    cv_f1_scores = {}
    for name, model in models.items():
        # cross_val_score re-fits the model internally; it does NOT touch test data
        cv_scores = cross_val_score(
            model, X_train_tfidf, y_train,
            cv=5, scoring="f1", n_jobs=-1,
        )
        cv_f1_scores[name] = cv_scores.mean()
        print(f"  {name:<25} CV F1 = {cv_scores.mean():.4f}"
              f"  (+/- {cv_scores.std():.4f})")

    # Select by cross-validated F1
    best_name    = max(cv_f1_scores, key=cv_f1_scores.get)
    best_model   = models[best_name]
    best_metrics = results[best_name]

    # Also store cv_f1 in results for display
    for name in results:
        results[name]["cv_f1"] = round(cv_f1_scores[name], 4)

    print(f"\n  >> Best model (by CV F1): {best_name}  "
          f"(CV F1 = {cv_f1_scores[best_name]:.4f})")

    # Print final test-set comparison table
    print(f"\n  {'Model':<25} {'CV F1':>8} {'Test F1':>9} {'Accuracy':>10}"
          f" {'Precision':>10} {'Recall':>8}")
    print("  " + "-"*75)
    for name, m in results.items():
        marker = "  <<" if name == best_name else ""
        print(f"  {name:<25} {cv_f1_scores[name]:>8.4f} {m['f1']:>9.4f}"
              f" {m['accuracy']:>10.4f} {m['precision']:>10.4f}"
              f" {m['recall']:>8.4f}{marker}")

    return best_name, best_model, best_metrics


# ──────────────────────────────────────────────
# 8. Save model & vectoriser
# ──────────────────────────────────────────────
def save_artifacts(model, vectorizer, best_name: str, all_results: dict,
                   dataset_info: dict):
    print("\n" + "="*60)
    print("  STEP 7 - Saving Model & Vectoriser")
    print("="*60)

    joblib.dump(model,      MODEL_PATH)
    joblib.dump(vectorizer, VECTOR_PATH)

    # Persist metrics so app.py can load them without retraining
    serialisable = {
        name: {k: v for k, v in m.items() if k != "y_pred"}
        for name, m in all_results.items()
    }
    payload = {
        "best_model_name": best_name,
        "model_results"  : serialisable,
        "dataset_info"   : dataset_info,
    }
    with open(METRICS_PATH, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"  Model saved     : {MODEL_PATH}")
    print(f"  Vectoriser saved: {VECTOR_PATH}")
    print(f"  Metrics saved   : {METRICS_PATH}")


# ──────────────────────────────────────────────
# 9. Generate plots
# ──────────────────────────────────────────────
def plot_class_distribution(df: pd.DataFrame):
    counts = df["label"].value_counts().sort_index()
    labels = ["HAM (0)", "SPAM (1)"]
    colors = ["#3b82d4", "#e05252"]

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(labels, [counts.get(0, 0), counts.get(1, 0)],
                  color=colors, edgecolor="white", width=0.5)
    for bar, val in zip(bars, [counts.get(0, 0), counts.get(1, 0)]):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 20, str(val),
                ha="center", va="bottom", fontsize=11, fontweight="bold")

    ax.set_title("Class Distribution (HAM vs SPAM)", fontsize=13, pad=12)
    ax.set_ylabel("Number of Emails", fontsize=11)
    ax.set_ylim(0, max(counts) * 1.15)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    path = os.path.join(OUTPUTS_DIR, "class_distribution.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_model_comparison(results: dict):
    names  = list(results.keys())
    f1s    = [results[n]["f1"]        for n in names]
    accs   = [results[n]["accuracy"]  for n in names]
    precs  = [results[n]["precision"] for n in names]
    recs   = [results[n]["recall"]    for n in names]

    x = np.arange(len(names))
    width = 0.2

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - 1.5*width, accs,  width, label="Accuracy",  color="#3b82d4")
    ax.bar(x - 0.5*width, precs, width, label="Precision", color="#7c5cd8")
    ax.bar(x + 0.5*width, recs,  width, label="Recall",    color="#22c55e")
    ax.bar(x + 1.5*width, f1s,   width, label="F1 Score",  color="#f97316")

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=10)
    # y-axis from 0 so differences are not visually exaggerated.
    # Values are annotated on each bar so close scores are still readable.
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title("Model Performance Comparison (y-axis: 0 to 1)", fontsize=13, pad=12)
    ax.legend(fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)

    # Annotate bar values
    for bars_x, vals in zip(
        [x - 1.5*width, x - 0.5*width, x + 0.5*width, x + 1.5*width],
        [accs, precs, recs, f1s],
    ):
        for bx, val in zip(bars_x, vals):
            ax.text(bx, val + 0.008, f"{val:.3f}",
                    ha="center", va="bottom", fontsize=7, color="#334155")

    fig.tight_layout()
    path = os.path.join(OUTPUTS_DIR, "model_comparison.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_confusion_matrix(y_test, y_pred, model_name: str):
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["HAM", "SPAM"],
    )
    fig, ax = plt.subplots(figsize=(5, 4))
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(f"Confusion Matrix - {model_name}", fontsize=12, pad=10)
    fig.tight_layout()
    path = os.path.join(OUTPUTS_DIR, "confusion_matrix.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"  Saved: {path}")


# ──────────────────────────────────────────────
# main
# ──────────────────────────────────────────────
def main():
    print("\n" + "="*60)
    print("  Spam Email Detection - Model Training Pipeline")
    print("="*60)

    # 1. Load
    df_raw = load_and_inspect(DATA_PATH)

    # 2. Clean
    df = prepare_data(df_raw)

    # Collect dataset stats for app.py dashboard
    dataset_info = {
        "total"   : int(len(df)),
        "ham"     : int((df["label"] == 0).sum()),
        "spam"    : int((df["label"] == 1).sum()),
    }

    # 3. Split
    X_train, X_test, y_train, y_test = split_data(df)
    dataset_info["train"] = int(len(X_train))
    dataset_info["test"]  = int(len(X_test))

    # 4. TF-IDF
    vectorizer, X_train_tfidf, X_test_tfidf = build_tfidf_features(X_train, X_test)

    # 5. Train all models
    models  = get_models()
    results = train_and_compare(models, X_train_tfidf, X_test_tfidf, y_train, y_test)

    # 6. Select best via cross-validation (avoids test-set selection bias)
    best_name, best_model, best_metrics = select_best_model(
        models, results, X_train_tfidf, y_train
    )
    # Re-fit best model on the full training set after CV-based selection
    best_model.fit(X_train_tfidf, y_train)

    # 7. Save
    save_artifacts(best_model, vectorizer, best_name, results, dataset_info)

    # 8. Plots
    print("\n" + "="*60)
    print("  STEP 8 - Generating Plots")
    print("="*60)
    plot_class_distribution(df)
    plot_model_comparison(results)
    plot_confusion_matrix(y_test, results[best_name]["y_pred"], best_name)

    # 9. Summary
    print("\n" + "="*60)
    print("  TRAINING COMPLETE - Summary")
    print("="*60)
    print(f"  Best model  : {best_name}")
    print(f"  Accuracy    : {best_metrics['accuracy']:.4f}")
    print(f"  Precision   : {best_metrics['precision']:.4f}")
    print(f"  Recall      : {best_metrics['recall']:.4f}")
    print(f"  F1 Score    : {best_metrics['f1']:.4f}")
    print(f"\n  Artefacts written to:")
    print(f"    {MODEL_PATH}")
    print(f"    {VECTOR_PATH}")
    print(f"    {METRICS_PATH}")
    print(f"\n  Run the app with:  streamlit run app.py")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
