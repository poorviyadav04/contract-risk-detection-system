import json
import re
from pathlib import Path
from collections import Counter
from datasets import load_dataset

"""
Phase 1 — Dataset Engineering
Loads LexGLUE unfair_tos, applies 4 engineering steps,
converts to instruction format, saves train/val/test JSONL files.
"""

OUTPUT_DIR = Path("data/processed")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

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

LABEL_EXPLANATIONS = {
    0: "it entirely removes the vendor's financial responsibility for damages caused by their product or service",
    1: "it allows the vendor to end your account or agreement at any time without notice or reason",
    2: "it allows the vendor to change the terms of the agreement without your explicit consent",
    3: "it gives the vendor the right to remove your content or data at their sole discretion",
    4: "it treats your continued use of the service as automatic acceptance of all terms, including future changes",
    5: "it forces disputes to be resolved under a specific jurisdiction's law, often unfavourable to the user",
    6: "it requires all legal disputes to be filed in a specific location, making legal action costly for the user",
    7: "it waives your right to sue in court and forces disputes into private arbitration favourable to the vendor",
}


# ── Step 1: Load ──────────────────────────────────────────────────────────────

def load_split(split: str):
    print(f"  Loading {split} split...")
    return load_dataset("lex_glue", "unfair_tos", split=split)


# ── Step 2: Filter trivial samples ───────────────────────────────────────────

def filter_trivial(dataset):
    before = len(dataset)
    dataset = dataset.filter(lambda x: len(x["text"].strip()) >= 30)
    print(f"  Trivial filter: {before} → {len(dataset)} examples")
    return dataset


# ── Step 3: Token length filter ───────────────────────────────────────────────

def filter_by_tokens(dataset, max_tokens=512):
    before = len(dataset)

    def within_limit(example):
        # Rough estimate: 4 chars per token
        return len(example["text"]) // 4 <= max_tokens

    dataset = dataset.filter(within_limit)
    print(f"  Token filter  : {before} → {len(dataset)} examples")
    return dataset


# ── Step 4: Convert to instruction format ────────────────────────────────────

def format_example(example):
    """
    Converts a raw (text, labels) pair into a rich instruction-output pair.
    Output includes verdict, risk type names, and a plain-English explanation.
    """
    clause = example["text"].strip()
    labels = example["labels"]

    instruction = (
        "Analyse the following contract clause. "
        "Identify whether it is fair or unfair. "
        "If unfair, name every risk type present and explain clearly "
        "why each risk is problematic for the user."
    )

    if not labels:
        output = (
            "VERDICT: FAIR\n\n"
            "This clause does not contain any identified unfair terms. "
            "It appears to establish standard contractual obligations "
            "without disproportionately favouring either party. "
            "You may still want a legal professional to review it in "
            "the context of the full agreement."
        )
    else:
        risk_names = [LABEL_MAP[l] for l in labels]
        risks_str  = ", ".join(risk_names)

        explanations = []
        for l in labels:
            explanations.append(
                f"- {LABEL_MAP[l]}: This clause is problematic because "
                f"{LABEL_EXPLANATIONS[l]}."
            )
        explanation_block = "\n".join(explanations)

        output = (
            f"VERDICT: UNFAIR\n\n"
            f"RISK TYPE(S): {risks_str}\n\n"
            f"EXPLANATION:\n{explanation_block}\n\n"
            f"RECOMMENDATION: Do not sign this clause without seeking "
            f"legal advice or negotiating a revision."
        )

    return {
        "instruction": instruction,
        "input":       clause,
        "output":      output,
    }


# ── Save ──────────────────────────────────────────────────────────────────────

def save_jsonl(examples: list, path: Path):
    with open(path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    print(f"  Saved {len(examples):,} examples → {path}")


# ── Label distribution report ─────────────────────────────────────────────────

def print_label_stats(dataset, split_name: str):
    all_labels = []
    fair_count = 0
    for ex in dataset:
        if not ex["labels"]:
            fair_count += 1
        else:
            all_labels.extend(ex["labels"])

    print(f"\n  {split_name} label distribution:")
    print(f"    FAIR (no risk): {fair_count}")
    counts = Counter(all_labels)
    for label_id, count in sorted(counts.items()):
        print(f"    {LABEL_MAP[label_id]}: {count}")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    splits = ["train", "validation", "test"]
    output_names = {
        "train":      "contract_train.jsonl",
        "validation": "contract_val.jsonl",
        "test":       "contract_test.jsonl",
    }

    for split in splits:
        print(f"\n{'='*50}")
        print(f"Processing: {split}")
        print(f"{'='*50}")

        ds = load_split(split)
        print(f"  Raw examples  : {len(ds)}")

        # Show label stats before filtering
        print_label_stats(ds, split)

        # Apply filters
        ds = filter_trivial(ds)
        ds = filter_by_tokens(ds)

        # Convert to instruction format
        formatted = [format_example(ex) for ex in ds]

        # Save
        out_path = OUTPUT_DIR / output_names[split]
        save_jsonl(formatted, out_path)

    print(f"\n{'='*50}")
    print("ALL SPLITS DONE")
    print(f"{'='*50}")
    print("\nFiles saved:")
    for name in output_names.values():
        p = OUTPUT_DIR / name
        print(f"  {p}  ({p.stat().st_size // 1024} KB)")

    print("\nNext: run verify_dataset.py on contract_train.jsonl")