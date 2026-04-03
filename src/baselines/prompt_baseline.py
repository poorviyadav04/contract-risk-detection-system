import json
import re
import requests
import numpy as np
from pathlib import Path
from sklearn.metrics import f1_score, classification_report

"""
Baseline 3 — Mistral 7B zero-shot via Ollama.
No training. Tests how much the base LLM knows without fine-tuning.
Runs on 200 test examples to keep it fast (~20 mins on CPU).
"""

OLLAMA_URL  = "http://localhost:11434/api/generate"
MODEL_NAME  = "mistral"
NUM_SAMPLES = 200          # subset of test set — full set would take 2+ hours
DATA_DIR    = Path("data/processed")
RESULTS_DIR = Path("data/results")
RESULTS_DIR.mkdir(exist_ok=True)

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

SYSTEM_PROMPT = """You are a legal contract analyst. 
Your job is to identify unfair clauses in contracts.

For each clause given, respond ONLY in this exact format:
VERDICT: FAIR or UNFAIR
RISK TYPE: (comma-separated list from: Limitation of Liability, Unilateral Termination, Unilateral Change, Content Removal, Contract by Using, Choice of Law, Choice of Venue, Forced Arbitration) — write None if FAIR

Do not write anything else. No explanation. Just the two lines."""


def load_jsonl(path: Path, max_examples: int = None):
    texts, labels = [], []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_examples and i >= max_examples:
                break
            ex = json.loads(line)
            texts.append(ex["input"])
            output = ex["output"]
            if output.startswith("VERDICT: FAIR"):
                labels.append([0.0] * 8)
            else:
                row = [0.0] * 8
                for lid, lname in LABEL_MAP.items():
                    if lname in output:
                        row[lid] = 1.0
                labels.append(row)
    return texts, labels


def query_ollama(clause: str) -> str:
    """Send a clause to Mistral via Ollama and get raw response."""
    prompt = f"{SYSTEM_PROMPT}\n\nClause:\n{clause}\n\nRespond now:"
    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": MODEL_NAME,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.0,   # deterministic
                    "num_predict": 80,    # short response only
                }
            },
            timeout=60,
        )
        return response.json().get("response", "").strip()
    except Exception as e:
        print(f"  Ollama error: {e}")
        return ""


def parse_response(response: str) -> list:
    """Parse Mistral output into a binary label vector."""
    row = [0.0] * 8
    response_upper = response.upper()

    if "VERDICT: FAIR" in response_upper:
        return row  # all zeros = fair

    # Extract risk types from response
    for lid, lname in LABEL_MAP.items():
        if lname.upper() in response_upper:
            row[lid] = 1.0

    # If UNFAIR but no risk types parsed, flag all as uncertain
    if "VERDICT: UNFAIR" in response_upper and sum(row) == 0:
        # Mark first label as a catch-all to avoid silent misses
        row[0] = 1.0

    return row


def run():
    print("Loading test data...")
    test_texts, test_labels = load_jsonl(
        DATA_DIR / "contract_test.jsonl",
        max_examples=NUM_SAMPLES
    )
    print(f"  Running on {len(test_texts)} examples")

    # Check Ollama is running
    try:
        requests.get("http://localhost:11434", timeout=5)
        print("  Ollama is running")
    except:
        print("ERROR: Ollama is not running.")
        print("Open a new terminal and run: ollama serve")
        return

    print(f"\nQuerying Mistral ({MODEL_NAME}) zero-shot...")
    print("This will take ~15-20 minutes on CPU. Progress below:\n")

    preds  = []
    labels = []

    for i, (clause, label) in enumerate(zip(test_texts, test_labels)):
        if i % 20 == 0:
            print(f"  [{i}/{len(test_texts)}] processing...")

        response = query_ollama(clause)
        pred     = parse_response(response)
        preds.append(pred)
        labels.append(label)

    preds  = np.array(preds)
    labels = np.array(labels)

    macro_f1 = f1_score(labels, preds, average="macro", zero_division=0)
    micro_f1 = f1_score(labels, preds, average="micro", zero_division=0)

    print(f"\n{'='*50}")
    print("RESULTS — Mistral 7B zero-shot (prompt-only)")
    print(f"{'='*50}")
    print(f"Macro F1 : {macro_f1:.4f}")
    print(f"Micro F1 : {micro_f1:.4f}")
    print(f"\nPer-class report:")
    print(classification_report(
        labels, preds,
        target_names=[LABEL_MAP[i] for i in range(8)],
        zero_division=0,
    ))

    # Update results.md
    with open(RESULTS_DIR / "results.md", "w") as f:
        f.write("# Baseline Results\n\n")
        f.write("| Model | Macro F1 |\n|---|---|\n")
        f.write("| Logistic Regression + TF-IDF | 0.1603 |\n")
        f.write("| BERT-base + weighted loss    | 0.5088 |\n")
        f.write(f"| Mistral 7B prompt-only       | {macro_f1:.4f} |\n")
        f.write("| Mistral 7B QLoRA fine-tuned  | TBD |\n")

    print(f"\nResults saved to data/results/results.md")
    print(f"Macro F1 = {macro_f1:.4f}")


if __name__ == "__main__":
    run()