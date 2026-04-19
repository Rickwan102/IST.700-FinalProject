"""
analyze_results.py

Analyzes experiment results and produces:
  1. Summary statistics by source tier and ground truth label
  2. Score delta (blind vs source-revealed) analysis
  3. Bias rate: misleading headlines rated >=4 with prestige source
  4. Statistical significance tests (paired t-test, Wilcoxon)
  5. Visualizations saved to results/figures/

Usage:
    python scripts/analyze_results.py
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

RAW_RESULTS_PATH = "results/raw_outputs.csv"
BASELINE_PATH = "results/baseline_results.csv"
FIGURES_DIR = "results/figures"
SUMMARY_OUTPUT = "results/summary_stats.csv"

TIER_LABELS = {
    None: "Blind",
    1: "Tier 1\n(High Prestige)",
    2: "Tier 2\n(Mid-Tier)",
    3: "Tier 3\n(Fringe)",
    4: "Tier 4\n(Fabricated)"
}

sns.set_theme(style="whitegrid", palette="muted")


def load_data():
    df = pd.read_csv(RAW_RESULTS_PATH)
    df["source_tier"] = df["source_tier"].where(df["condition"] == "source_revealed", other=None)
    df = df.dropna(subset=["rating"])
    df["rating"] = df["rating"].astype(int)
    print(f"Loaded {len(df)} rows | {df['rating'].isna().sum()} parse failures dropped")
    return df


def compute_score_deltas(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each headline, compute the delta between source-revealed and blind rating.
    Returns a DataFrame with one row per (headline_id, tier) pair.
    """
    blind = df[df["condition"] == "blind"][["headline_id", "rating", "ground_truth"]].rename(
        columns={"rating": "blind_rating"}
    )
    source = df[df["condition"] == "source_revealed"][
        ["headline_id", "source_tier", "source", "rating", "ground_truth"]
    ].rename(columns={"rating": "source_rating"})

    merged = source.merge(blind, on=["headline_id", "ground_truth"])
    merged["delta"] = merged["source_rating"] - merged["blind_rating"]
    return merged


def significance_tests(deltas: pd.DataFrame):
    """Run paired statistical tests per tier."""
    print("\n" + "=" * 60)
    print("STATISTICAL SIGNIFICANCE TESTS (vs Blind baseline)")
    print("=" * 60)

    for tier in sorted(deltas["source_tier"].unique()):
        tier_data = deltas[deltas["source_tier"] == tier]["delta"].dropna()
        if len(tier_data) < 5:
            continue

        t_stat, t_p = stats.ttest_1samp(tier_data, popmean=0)
        w_stat, w_p = stats.wilcoxon(tier_data) if len(tier_data) >= 10 else (None, None)

        print(f"\nTier {int(tier)}:")
        print(f"  Mean delta:      {tier_data.mean():+.3f}")
        print(f"  Median delta:    {tier_data.median():+.3f}")
        print(f"  Paired t-test:   t={t_stat:.3f}, p={t_p:.4f} {'*' if t_p < 0.05 else ''}")
        if w_p is not None:
            print(f"  Wilcoxon test:   W={w_stat:.1f}, p={w_p:.4f} {'*' if w_p < 0.05 else ''}")


def plot_mean_rating_by_tier(df: pd.DataFrame, figures_dir: str):
    """Bar chart: mean credibility rating by condition/tier, split by ground truth."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    for ax, label in zip(axes, ["credible", "misleading"]):
        subset = df[df["ground_truth"] == label].copy()
        subset["tier_label"] = subset["source_tier"].map(TIER_LABELS).fillna("Blind")

        order = ["Blind"] + [TIER_LABELS[i] for i in [1, 2, 3, 4] if TIER_LABELS[i] in subset["tier_label"].values]

        sns.barplot(
            data=subset,
            x="tier_label",
            y="rating",
            order=order,
            ax=ax,
            capsize=0.1,
            errwidth=1.5
        )
        ax.set_title(f"{'Credible' if label == 'credible' else 'Misleading'} Headlines", fontsize=13)
        ax.set_xlabel("Condition / Source Tier")
        ax.set_ylabel("Mean Credibility Rating (1–5)")
        ax.set_ylim(1, 5)
        ax.tick_params(axis="x", labelsize=9)

    fig.suptitle("Mean LLM Credibility Rating by Source Tier and Ground Truth", fontsize=14, y=1.02)
    plt.tight_layout()
    path = os.path.join(figures_dir, "mean_rating_by_tier.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_score_delta(deltas: pd.DataFrame, figures_dir: str):
    """Box plot: score delta (source - blind) by tier and ground truth."""
    deltas["tier_label"] = deltas["source_tier"].apply(lambda x: f"Tier {int(x)}")

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.boxplot(
        data=deltas,
        x="tier_label",
        y="delta",
        hue="ground_truth",
        palette={"credible": "#4C9BE8", "misleading": "#E8694C"},
        ax=ax
    )
    ax.axhline(0, color="black", linestyle="--", linewidth=1, alpha=0.6)
    ax.set_title("Credibility Score Delta: Source-Revealed vs. Blind", fontsize=14)
    ax.set_xlabel("Source Tier")
    ax.set_ylabel("Rating Delta (Source − Blind)")
    ax.legend(title="Ground Truth")

    path = os.path.join(figures_dir, "score_delta_by_tier.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_bias_rate(deltas: pd.DataFrame, figures_dir: str):
    """
    Bar chart: % of misleading headlines rated >=4 when source is revealed,
    vs when blind. This is the 'prestige laundering' bias rate.
    """
    misleading = deltas[deltas["ground_truth"] == "misleading"].copy()
    blind_df = pd.read_csv(RAW_RESULTS_PATH)
    blind_df = blind_df[(blind_df["condition"] == "blind") & (blind_df["ground_truth"] == "misleading")]
    blind_rate = (blind_df["rating"] >= 4).mean()

    tier_rates = (
        misleading.groupby("source_tier")
        .apply(lambda g: (g["source_rating"] >= 4).mean())
        .reset_index(name="bias_rate")
    )
    tier_rates["tier_label"] = tier_rates["source_tier"].apply(lambda x: f"Tier {int(x)}")

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(tier_rates["tier_label"], tier_rates["bias_rate"], color="#E8694C", alpha=0.8)
    ax.axhline(blind_rate, color="steelblue", linestyle="--", linewidth=1.5, label=f"Blind baseline ({blind_rate:.1%})")
    ax.set_title("'Prestige Laundering' Rate:\n% Misleading Headlines Rated ≥4 by Source Tier", fontsize=13)
    ax.set_ylabel("Rate of Misleading Headlines Rated ≥4")
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax.legend()

    for bar, val in zip(bars, tier_rates["bias_rate"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.1%}", ha="center", va="bottom", fontsize=10)

    path = os.path.join(figures_dir, "bias_rate.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def save_summary(df: pd.DataFrame, deltas: pd.DataFrame):
    """Save summary statistics table."""
    summary_rows = []

    # Blind condition
    blind = df[df["condition"] == "blind"]
    for label in ["credible", "misleading"]:
        subset = blind[blind["ground_truth"] == label]
        summary_rows.append({
            "condition": "Blind",
            "source_tier": None,
            "ground_truth": label,
            "n": len(subset),
            "mean_rating": round(subset["rating"].mean(), 3),
            "std_rating": round(subset["rating"].std(), 3),
            "mean_delta": None
        })

    # Source-revealed conditions
    for tier in sorted(df["source_tier"].dropna().unique()):
        for label in ["credible", "misleading"]:
            subset = df[(df["source_tier"] == tier) & (df["ground_truth"] == label)]
            delta_subset = deltas[(deltas["source_tier"] == tier) & (deltas["ground_truth"] == label)]
            summary_rows.append({
                "condition": f"Source-Revealed",
                "source_tier": int(tier),
                "ground_truth": label,
                "n": len(subset),
                "mean_rating": round(subset["rating"].mean(), 3),
                "std_rating": round(subset["rating"].std(), 3),
                "mean_delta": round(delta_subset["delta"].mean(), 3) if len(delta_subset) > 0 else None
            })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(SUMMARY_OUTPUT, index=False)
    print(f"\n✓ Summary stats saved to {SUMMARY_OUTPUT}")
    print(summary_df.to_string(index=False))


def main():
    os.makedirs(FIGURES_DIR, exist_ok=True)

    df = load_data()
    deltas = compute_score_deltas(df)

    print("\nGenerating visualizations...")
    plot_mean_rating_by_tier(df, FIGURES_DIR)
    plot_score_delta(deltas, FIGURES_DIR)
    plot_bias_rate(deltas, FIGURES_DIR)

    significance_tests(deltas)
    save_summary(df, deltas)

    print("\n✓ Analysis complete.")


if __name__ == "__main__":
    main()
