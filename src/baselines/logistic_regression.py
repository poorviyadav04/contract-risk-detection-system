import json
import numpy as np
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.preprocessing import MultiLabelBinarizer
import mlflow

"""
Baseline 1 — Logistic Regression on TF-IDF features.
Multi-label classification: one binary classifier per risk label.
Runs fully on CPU in under 2 minutes.
"""

LABEL_MAP = {
    0: "Limitation of Liability",
    1: "Unilateral Termination",
    2: "Unilateral Change",
    3: "Content Removal",
    4: "Contract by Using",
    5: "Choice of Law",
    6: "Choice of Venue",
    7: "Forced Arbitration",
}

DATA_DIR = Path("data/processed")


# ── Load JSONL back to (text, labels) pairs ───────────────────────────────────

def load_jsonl(path: Path):
    texts, labels = [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            ex = json.loads(line)
            texts.append(ex["input"])

            # Parse labels back from output text
            output = ex["output"]
            if output.startswith("VERDICT: FAIR"):
                labels.append([])
            else:
                found = []
                for lid, lname in LABEL_MAP.items():
                    if lname in output:
                        found.append(lid)
                labels.append(found)

    return texts, labels


def run():
    print("Loading data...")
    train_texts, train_labels = load_jsonl(DATA_DIR / "contract_train.jsonl")
    test_texts,  test_labels  = load_jsonl(DATA_DIR / "contract_test.jsonl")

    print(f"  Train: {len(train_texts):,} examples")
    print(f"  Test : {len(test_texts):,} examples")

    # Binarize labels for multi-label classification
    mlb = MultiLabelBinarizer(classes=list(range(8)))
    y_train = mlb.fit_transform(train_labels)
    y_test  = mlb.transform(test_labels)

    # TF-IDF features
    print("\nBuilding TF-IDF features...")
    vectorizer = TfidfVectorizer(
        max_features=15000,
        ngram_range=(1, 2),    # unigrams + bigrams
        sublinear_tf=True,     # log TF scaling
        min_df=2,              # ignore very rare terms
    )
    X_train = vectorizer.fit_transform(train_texts)
    X_test  = vectorizer.transform(test_texts)
    print(f"  Feature matrix: {X_train.shape}")

    # Train one LR classifier per label (One-vs-Rest)
    print("\nTraining Logistic Regression (One-vs-Rest)...")
    clf = LogisticRegression(
        max_iter=1000,
        C=1.0,
        solver="lbfgs",
    )

    # Fit per label
    from sklearn.multiclass import OneVsRestClassifier
    ovr = OneVsRestClassifier(clf)
    ovr.fit(X_train, y_train)

    # Evaluate
    print("\nEvaluating...")
    y_pred = ovr.predict(X_test)

    macro_f1  = f1_score(y_test, y_pred, average="macro",  zero_division=0)
    micro_f1  = f1_score(y_test, y_pred, average="micro",  zero_division=0)
    sample_f1 = f1_score(y_test, y_pred, average="samples",zero_division=0)

    print(f"\n{'='*50}")
    print("RESULTS — Logistic Regression + TF-IDF")
    print(f"{'='*50}")
    print(f"Macro  F1 : {macro_f1:.4f}")
    print(f"Micro  F1 : {micro_f1:.4f}")
    print(f"Sample F1 : {sample_f1:.4f}")

    print("\nPer-class report:")
    print(classification_report(
        y_test, y_pred,
        target_names=[LABEL_MAP[i] for i in range(8)],
        zero_division=0
    ))

    # Log to MLflow
    print("Logging to MLflow...")
    mlflow.set_experiment("contract-risk-baselines")
    with mlflow.start_run(run_name="logistic_regression_tfidf"):
        mlflow.log_param("model",        "LogisticRegression")
        mlflow.log_param("features",     "TF-IDF")
        mlflow.log_param("max_features", 15000)
        mlflow.log_param("ngram_range",  "(1,2)")
        mlflow.log_param("C",            1.0)
        mlflow.log_metric("macro_f1",    macro_f1)
        mlflow.log_metric("micro_f1",    micro_f1)
        mlflow.log_metric("sample_f1",   sample_f1)

        # Per-class F1
        per_class = f1_score(y_test, y_pred, average=None, zero_division=0)
        for i, score in enumerate(per_class):
            mlflow.log_metric(f"f1_{LABEL_MAP[i].replace(' ','_')}", score)

    print(f"\nMLflow run logged. Macro F1 = {macro_f1:.4f}")
    print("Run 'mlflow ui' in terminal to view results.")

    # Save result summary
    results_dir = Path("data/results")
    results_dir.mkdir(exist_ok=True)
    with open(results_dir / "results.md", "w") as f:
        f.write("# Baseline Results\n\n")
        f.write("| Model | Macro F1 |\n")
        f.write("|---|---|\n")
        f.write(f"| Logistic Regression + TF-IDF | {macro_f1:.4f} |\n")
        f.write("| BERT-base (fine-tuned) | TBD |\n")
        f.write("| Mistral 7B prompt-only | TBD |\n")
        f.write("| Mistral 7B QLoRA fine-tuned | TBD |\n")

    print(f"Results saved to data/results/results.md")


if __name__ == "__main__":
    run()