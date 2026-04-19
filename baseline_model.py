"""
baseline_model.py

Logistic regression baseline classifier using TF-IDF features on headline text.
Uses only headline content — no source information — to classify credible vs misleading.

Outputs:
    - Prints classification report and accuracy
    - Appends baseline predictions to results/raw_outputs.csv for comparison

Usage:
    python scripts/baseline_model.py
"""

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, accuracy_score
from sklearn.pipeline import Pipeline
import json
import os

DATA_PATH = "data/fin_fact_sample.csv"
OUTPUT_PATH = "results/baseline_results.csv"
RANDOM_SEED = 42
CV_FOLDS = 5


def run_baseline():
    os.makedirs("results", exist_ok=True)

    # Load data
    df = pd.read_csv(DATA_PATH, index_col="headline_id")
    print(f"Loaded {len(df)} headlines")
    print(f"Label distribution:\n{df['binary_label'].value_counts()}\n")

    X = df["headline"]
    y = (df["binary_label"] == "credible").astype(int)  # 1 = credible, 0 = misleading

    # ── Build pipeline ────────────────────────────────────────────────────────
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2),      # unigrams + bigrams
            max_features=10000,
            stop_words="english",
            sublinear_tf=True
        )),
        ("clf", LogisticRegression(
            max_iter=1000,
            random_state=RANDOM_SEED,
            C=1.0
        ))
    ])

    # ── Cross-validation ──────────────────────────────────────────────────────
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    cv_scores = cross_val_score(pipeline, X, y, cv=cv, scoring="accuracy")

    print("=" * 50)
    print("BASELINE MODEL: Logistic Regression + TF-IDF")
    print("=" * 50)
    print(f"Cross-validation accuracy ({CV_FOLDS}-fold): {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
    print(f"Individual fold scores: {[round(s, 3) for s in cv_scores]}\n")

    # ── Fit on full data and get predictions ──────────────────────────────────
    pipeline.fit(X, y)
    y_pred = pipeline.predict(X)
    y_prob = pipeline.predict_proba(X)[:, 1]

    print("Classification Report (full dataset, for reference):")
    print(classification_report(y, y_pred, target_names=["misleading", "credible"]))

    # ── Top TF-IDF features ───────────────────────────────────────────────────
    vectorizer = pipeline.named_steps["tfidf"]
    clf = pipeline.named_steps["clf"]
    feature_names = vectorizer.get_feature_names_out()
    coefs = clf.coef_[0]

    top_credible = sorted(zip(coefs, feature_names), reverse=True)[:15]
    top_misleading = sorted(zip(coefs, feature_names))[:15]

    print("\nTop features predicting CREDIBLE:")
    for coef, feat in top_credible:
        print(f"  {feat:30s} {coef:+.3f}")

    print("\nTop features predicting MISLEADING:")
    for coef, feat in top_misleading:
        print(f"  {feat:30s} {coef:+.3f}")

    # ── Save predictions ──────────────────────────────────────────────────────
    df["baseline_pred"] = ["credible" if p == 1 else "misleading" for p in y_pred]
    df["baseline_prob_credible"] = y_prob
    df["baseline_correct"] = (df["binary_label"] == df["baseline_pred"])

    df.to_csv(OUTPUT_PATH)
    print(f"\n✓ Baseline results saved to {OUTPUT_PATH}")

    # ── Summary stats ─────────────────────────────────────────────────────────
    summary = {
        "model": "Logistic Regression + TF-IDF",
        "cv_folds": CV_FOLDS,
        "cv_accuracy_mean": round(cv_scores.mean(), 4),
        "cv_accuracy_std": round(cv_scores.std(), 4),
        "full_dataset_accuracy": round(accuracy_score(y, y_pred), 4)
    }
    with open("results/baseline_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Summary saved to results/baseline_summary.json")


if __name__ == "__main__":
    run_baseline()
