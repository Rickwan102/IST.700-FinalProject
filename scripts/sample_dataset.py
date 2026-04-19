"""
sample_dataset.py

Downloads and samples headlines from the Fin-Fact dataset.
Produces data/fin_fact_sample.csv with balanced credible/misleading headlines.

Usage:
    python scripts/sample_dataset.py
"""

import pandas as pd
import json
import random
import os

# ── Configuration ────────────────────────────────────────────────────────────
SAMPLE_SIZE = 150          # Total headlines to sample
BALANCE_RATIO = 0.5        # 50% credible, 50% misleading
RANDOM_SEED = 42
OUTPUT_PATH = "data/fin_fact_sample.csv"

# Fin-Fact dataset URL (raw CSV from GitHub)
# Adjust this path if you've downloaded the dataset locally
FIN_FACT_URL = "https://raw.githubusercontent.com/IIT-DM/Fin-Fact/main/Fin-Fact.csv"


def load_fin_fact(url: str) -> pd.DataFrame:
    """Load the Fin-Fact dataset from GitHub or local path."""
    print(f"Loading Fin-Fact dataset from: {url}")
    try:
        df = pd.read_csv(url)
        print(f"  Loaded {len(df)} rows with columns: {list(df.columns)}")
        return df
    except Exception as e:
        raise RuntimeError(
            f"Could not load Fin-Fact dataset. "
            f"Please download it manually from https://github.com/IIT-DM/Fin-Fact "
            f"and place it at data/Fin-Fact.csv\n\nError: {e}"
        )


def prepare_sample(df: pd.DataFrame, sample_size: int, balance_ratio: float, seed: int) -> pd.DataFrame:
    """
    Sample headlines from the dataset, balanced between credible and misleading.

    Fin-Fact label column is expected to be 'label' or 'verdict' with values
    like 'true', 'false', 'misleading', etc. Adjust column names below if needed.
    """
    # ── Identify label column ────────────────────────────────────────────────
    # Common label column names in Fin-Fact — adjust if dataset differs
    label_col = None
    for candidate in ["label", "verdict", "claim_label", "fact_label"]:
        if candidate in df.columns:
            label_col = candidate
            break

    if label_col is None:
        print(f"  WARNING: Could not find label column. Available columns: {list(df.columns)}")
        print("  Please set label_col manually in this script.")
        label_col = df.columns[1]  # fallback to second column

    # ── Identify headline column ─────────────────────────────────────────────
    headline_col = None
    for candidate in ["claim", "headline", "title", "text"]:
        if candidate in df.columns:
            headline_col = candidate
            break

    if headline_col is None:
        print(f"  WARNING: Could not find headline column. Available columns: {list(df.columns)}")
        headline_col = df.columns[0]

    print(f"  Using label column: '{label_col}', headline column: '{headline_col}'")
    print(f"  Label distribution:\n{df[label_col].value_counts()}")

    # ── Map labels to binary credible/misleading ─────────────────────────────
    credible_keywords = ["true", "correct", "accurate", "credible", "mostly true"]
    misleading_keywords = ["false", "misleading", "incorrect", "fake", "mostly false", "pants on fire"]

    def map_label(val):
        val_lower = str(val).lower().strip()
        if any(kw in val_lower for kw in credible_keywords):
            return "credible"
        elif any(kw in val_lower for kw in misleading_keywords):
            return "misleading"
        return None

    df["binary_label"] = df[label_col].apply(map_label)
    df = df.dropna(subset=["binary_label"])

    # ── Sample balanced ──────────────────────────────────────────────────────
    n_credible = int(sample_size * balance_ratio)
    n_misleading = sample_size - n_credible

    credible_sample = df[df["binary_label"] == "credible"].sample(
        n=min(n_credible, len(df[df["binary_label"] == "credible"])),
        random_state=seed
    )
    misleading_sample = df[df["binary_label"] == "misleading"].sample(
        n=min(n_misleading, len(df[df["binary_label"] == "misleading"])),
        random_state=seed
    )

    sample = pd.concat([credible_sample, misleading_sample]).sample(frac=1, random_state=seed)

    result = sample[[headline_col, "binary_label"]].rename(columns={headline_col: "headline"})
    result = result.reset_index(drop=True)
    result.index.name = "headline_id"

    print(f"\n  Final sample: {len(result)} headlines")
    print(f"  {result['binary_label'].value_counts().to_dict()}")

    return result


def main():
    os.makedirs("data", exist_ok=True)
    random.seed(RANDOM_SEED)

    # Load dataset
    try:
        df = load_fin_fact(FIN_FACT_URL)
    except RuntimeError as e:
        print(e)
        return

    # Sample
    sample = prepare_sample(df, SAMPLE_SIZE, BALANCE_RATIO, RANDOM_SEED)

    # Save
    sample.to_csv(OUTPUT_PATH)
    print(f"\n✓ Saved sample to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
