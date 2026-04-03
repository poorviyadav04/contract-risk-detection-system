import json
import numpy as np
from pathlib import Path
from sklearn.metrics import f1_score, classification_report
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
import mlflow

"""
Baseline 2 — BERT-base fine-tuned with a multi-label classification head.
Runs on CPU — will take 30-40 minutes. Go make tea.
"""

MODEL_NAME = "bert-base-uncased"
MAX_LENGTH = 256
BATCH_SIZE = 16
NUM_EPOCHS = 3
NUM_LABELS = 8

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

DATA_DIR    = Path("data/processed")
RESULTS_DIR = Path("data/results")
RESULTS_DIR.mkdir(exist_ok=True)


# ── Dataset ───────────────────────────────────────────────────────────────────

def load_jsonl(path: Path):
    texts, labels = [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            ex = json.loads(line)
            texts.append(ex["input"])
            output = ex["output"]
            if output.startswith("VERDICT: FAIR"):
                labels.append([0.0] * NUM_LABELS)
            else:
                row = [0.0] * NUM_LABELS
                for lid, lname in LABEL_MAP.items():
                    if lname in output:
                        row[lid] = 1.0
                labels.append(row)
    return texts, labels


class ClauseDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )
        self.labels = torch.tensor(labels, dtype=torch.float)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item


# ── Custom trainer for multi-label BCE loss ───────────────────────────────────

class MultiLabelTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            logits, labels
        )
        return (loss, outputs) if return_outputs else loss


# ── Evaluation metric ─────────────────────────────────────────────────────────

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = (logits > 0.0).astype(int)   # threshold at 0
    macro_f1 = f1_score(labels, preds, average="macro", zero_division=0)
    micro_f1 = f1_score(labels, preds, average="micro", zero_division=0)
    return {"macro_f1": macro_f1, "micro_f1": micro_f1}


# ── Main ──────────────────────────────────────────────────────────────────────

def run():
    print(f"Loading tokenizer: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    print("Loading data...")
    train_texts, train_labels = load_jsonl(DATA_DIR / "contract_train.jsonl")
    val_texts,   val_labels   = load_jsonl(DATA_DIR / "contract_val.jsonl")
    test_texts,  test_labels  = load_jsonl(DATA_DIR / "contract_test.jsonl")

    print(f"  Train: {len(train_texts):,} | Val: {len(val_texts):,} | Test: {len(test_texts):,}")

    print("Tokenising...")
    train_ds = ClauseDataset(train_texts, train_labels, tokenizer)
    val_ds   = ClauseDataset(val_texts,   val_labels,   tokenizer)
    test_ds  = ClauseDataset(test_texts,  test_labels,  tokenizer)

    print(f"Loading model: {MODEL_NAME}")
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_LABELS,
        problem_type="multi_label_classification",
    )

    training_args = TrainingArguments(
        output_dir="data/bert_checkpoints",
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=2e-5,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        logging_steps=50,
        fp16=False,          # CPU only
        report_to="none",
    )

    trainer = MultiLabelTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
    )

    print("\nTraining BERT-base... (30-40 mins on CPU, go make tea)")
    trainer.train()

    # Evaluate on test set
    print("\nEvaluating on test set...")
    preds_output = trainer.predict(test_ds)
    logits = preds_output.predictions
    preds  = (logits > 0.0).astype(int)
    labels = np.array(test_labels)

    macro_f1 = f1_score(labels, preds, average="macro", zero_division=0)
    micro_f1 = f1_score(labels, preds, average="micro", zero_division=0)

    print(f"\n{'='*50}")
    print("RESULTS — BERT-base fine-tuned")
    print(f"{'='*50}")
    print(f"Macro F1 : {macro_f1:.4f}")
    print(f"Micro F1 : {micro_f1:.4f}")
    print("\nPer-class report:")
    print(classification_report(
        labels, preds,
        target_names=[LABEL_MAP[i] for i in range(8)],
        zero_division=0,
    ))

    # Log to MLflow
    mlflow.set_experiment("contract-risk-baselines")
    with mlflow.start_run(run_name="bert_base_finetuned"):
        mlflow.log_param("model",       MODEL_NAME)
        mlflow.log_param("max_length",  MAX_LENGTH)
        mlflow.log_param("epochs",      NUM_EPOCHS)
        mlflow.log_param("batch_size",  BATCH_SIZE)
        mlflow.log_param("lr",          2e-5)
        mlflow.log_metric("macro_f1",   macro_f1)
        mlflow.log_metric("micro_f1",   micro_f1)

        per_class = f1_score(labels, preds, average=None, zero_division=0)
        for i, score in enumerate(per_class):
            mlflow.log_metric(f"f1_{LABEL_MAP[i].replace(' ','_')}", score)

    print(f"\nMLflow run logged. Macro F1 = {macro_f1:.4f}")

    # Update results.md
    with open(RESULTS_DIR / "results.md", "w") as f:
        f.write("# Baseline Results\n\n")
        f.write("| Model | Macro F1 |\n")
        f.write("|---|---|\n")
        f.write(f"| Logistic Regression + TF-IDF | 0.1603 |\n")
        f.write(f"| BERT-base (fine-tuned) | {macro_f1:.4f} |\n")
        f.write("| Mistral 7B prompt-only | TBD |\n")
        f.write("| Mistral 7B QLoRA fine-tuned | TBD |\n")

    print("Results saved to data/results/results.md")


if __name__ == "__main__":
    run()