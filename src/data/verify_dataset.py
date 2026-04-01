import json
from pathlib import Path
from collections import Counter

# ── Load ──────────────────────────────────────────────────────────────────────

def load_jsonl(path: str) -> list:
    """Read every line of the JSONL file into a list of dicts."""
    examples = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                examples.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"  Bad JSON at line {i+1}: {e}")
    return examples


# ── Checks ────────────────────────────────────────────────────────────────────

def check_structure(examples: list) -> list:
    """Every example must have instruction, input, output keys."""
    bad = []
    for i, ex in enumerate(examples):
        missing = [k for k in ["instruction", "input", "output"] if k not in ex]
        if missing:
            bad.append((i, f"Missing keys: {missing}"))
    return bad


def check_lengths(examples: list, min_out=50, max_out=2000) -> list:
    """Flag examples where output is too short or suspiciously long."""
    bad = []
    for i, ex in enumerate(examples):
        out_len = len(str(ex.get("output", "")))
        if out_len < min_out:
            bad.append((i, f"Output too short: {out_len} chars"))
        if out_len > max_out:
            bad.append((i, f"Output very long: {out_len} chars — may exceed token limit"))
    return bad


def check_duplicates(examples: list) -> list:
    """Find exact duplicate instructions."""
    seen = Counter()
    for ex in examples:
        seen[ex.get("instruction", "").strip()] += 1
    dupes = [(instr[:80], count) for instr, count in seen.items() if count > 1]
    return dupes


def estimate_tokens(text: str) -> int:
    """Rough token estimate: ~4 chars per token for English."""
    return len(text) // 4


def check_token_lengths(examples: list, max_tokens=1024) -> list:
    """Flag examples that will likely exceed the model's max_seq_length."""
    over = []
    for i, ex in enumerate(examples):
        full_text = ex.get("instruction","") + ex.get("input","") + ex.get("output","")
        tokens = estimate_tokens(full_text)
        if tokens > max_tokens:
            over.append((i, f"~{tokens} tokens — will be truncated at {max_tokens}"))
    return over


# ── Report ────────────────────────────────────────────────────────────────────

def print_sample(examples: list, n=3):
    """Print a few examples so you can eyeball quality."""
    print(f"\n{'='*50}")
    print(f"SAMPLE EXAMPLES (first {n})")
    print(f"{'='*50}")
    for i, ex in enumerate(examples[:n]):
        print(f"\n--- Example {i+1} ---")
        print(f"INSTRUCTION: {ex['instruction'][:120]}")
        print(f"OUTPUT:      {ex['output'][:300]}")
        print()


if __name__ == "__main__":
    path = "data/processed/contract_train.jsonl"

    print(f"\nLoading: {path}")
    examples = load_jsonl(path)
    print(f"Total examples loaded: {len(examples)}")

    # Run all checks
    structure_errors = check_structure(examples)
    length_errors    = check_lengths(examples)
    duplicates       = check_duplicates(examples)
    token_warnings   = check_token_lengths(examples)

    # Print report
    print(f"\n{'='*50}")
    print("VALIDATION REPORT")
    print(f"{'='*50}")

    if structure_errors:
        print(f"\n❌ Structure errors ({len(structure_errors)}):")
        for idx, msg in structure_errors[:5]:
            print(f"   Line {idx+1}: {msg}")
    else:
        print(f"\n✅ Structure      : All {len(examples)} examples have correct keys")

    if length_errors:
        print(f"\n⚠️  Length issues ({len(length_errors)}):")
        for idx, msg in length_errors[:5]:
            print(f"   Example {idx+1}: {msg}")
    else:
        print(f"✅ Output lengths  : All within acceptable range")

    if duplicates:
        print(f"\n⚠️  Duplicates ({len(duplicates)}):")
        for instr, count in duplicates[:5]:
            print(f"   '{instr}...' appears {count}x")
    else:
        print(f"✅ Duplicates      : None found")

    if token_warnings:
        print(f"\n⚠️  Token length warnings ({len(token_warnings)}):")
        for idx, msg in token_warnings[:5]:
            print(f"   Example {idx+1}: {msg}")
    else:
        print(f"✅ Token lengths   : All examples fit within 1024 tokens")

    # Summary stats
    print(f"\n{'='*50}")
    print("SUMMARY STATS")
    print(f"{'='*50}")
    avg_out = sum(len(ex.get("output","")) for ex in examples) // len(examples)
    avg_tok = sum(estimate_tokens(ex.get("instruction","") + ex.get("output","")) for ex in examples) // len(examples)
    print(f"Total examples   : {len(examples)}")
    print(f"Avg output length: {avg_out} chars")
    print(f"Avg token estimate: ~{avg_tok} tokens per example")

    # Print samples
    print_sample(examples, n=2)

    # Check duplicates by input (clause text) — report only, do not overwrite
    seen = set()
    duplicates_found = 0
    for ex in examples:
        key = ex.get("input", "").strip()
        if key in seen:
            duplicates_found += 1
        else:
            seen.add(key)

    if duplicates_found > 0:
        print(f"\n⚠️  Duplicates: {duplicates_found} duplicate clauses found.")
    else:
        print(f"\n✅ Duplicates: None found.")

    print("\nDataset is ready for training.")