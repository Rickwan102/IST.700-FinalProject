"""
run_experiment.py

Main experiment runner. For each headline in the sample, runs:
  - Condition A: Blind rating (no source)
  - Condition B: Source-revealed rating (one source per tier)

Logs all results to results/raw_outputs.csv.

Usage:
    python scripts/run_experiment.py

Requirements:
    OPENAI_API_KEY environment variable must be set.
"""

import os
import re
import json
import time
import random
import pandas as pd
from openai import OpenAI
from datetime import datetime

# ── Configuration ────────────────────────────────────────────────────────────
MODEL = "gpt-4o"
TEMPERATURE = 0               # Keep deterministic
MAX_TOKENS = 150
SLEEP_BETWEEN_CALLS = 0.5     # Seconds — avoids rate limit errors
RANDOM_SEED = 42

DATA_PATH = "data/fin_fact_sample.csv"
SOURCES_PATH = "data/sources.json"
PROMPTS_PATH = "prompts/prompts.json"
OUTPUT_PATH = "results/raw_outputs.csv"

# One source to use per tier (selected randomly at runtime, logged for reproducibility)
SOURCES_PER_TIER = 1


# ── Load config files ─────────────────────────────────────────────────────────
def load_configs():
    with open(SOURCES_PATH) as f:
        sources_config = json.load(f)
    with open(PROMPTS_PATH) as f:
        prompts_config = json.load(f)
    return sources_config, prompts_config


def select_sources(sources_config: dict, seed: int) -> dict:
    """Pick one source per tier to use in the experiment."""
    random.seed(seed)
    selected = {}
    for tier, info in sources_config["tiers"].items():
        selected[tier] = random.choice(info["sources"])
    print("Selected sources per tier:")
    for tier, source in selected.items():
        label = sources_config["tiers"][tier]["label"]
        print(f"  Tier {tier} ({label}): {source}")
    return selected


# ── LLM call ─────────────────────────────────────────────────────────────────
def call_llm(client: OpenAI, system_prompt: str, user_prompt: str) -> str:
    """Call the OpenAI API and return raw text response."""
    response = client.chat.completions.create(
        model=MODEL,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )
    return response.choices[0].message.content.strip()


def parse_response(response_text: str) -> tuple[int | None, str]:
    """
    Extract rating (1-5) and justification from structured LLM response.
    Returns (rating, justification) or (None, raw_text) if parsing fails.
    """
    rating_match = re.search(r"RATING:\s*([1-5])", response_text)
    justification_match = re.search(r"JUSTIFICATION:\s*(.+)", response_text, re.DOTALL)

    rating = int(rating_match.group(1)) if rating_match else None
    justification = justification_match.group(1).strip() if justification_match else response_text

    return rating, justification


# ── Main experiment loop ──────────────────────────────────────────────────────
def run_experiment():
    # Setup
    os.makedirs("results", exist_ok=True)
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    sources_config, prompts_config = load_configs()
    selected_sources = select_sources(sources_config, RANDOM_SEED)

    headlines = pd.read_csv(DATA_PATH, index_col="headline_id")
    print(f"\nLoaded {len(headlines)} headlines from {DATA_PATH}")

    system_prompt = prompts_config["system_prompt"]
    blind_template = prompts_config["blind_prompt"]["template"]
    source_template = prompts_config["source_revealed_prompt"]["template"]

    results = []
    total_calls = len(headlines) * (1 + len(selected_sources))  # blind + one per tier
    call_count = 0

    print(f"\nStarting experiment: {total_calls} total LLM calls\n")

    for headline_id, row in headlines.iterrows():
        headline = row["headline"]
        ground_truth = row["binary_label"]

        # ── Condition A: Blind ────────────────────────────────────────────────
        blind_prompt = blind_template.format(headline=headline)
        try:
            response = call_llm(client, system_prompt, blind_prompt)
            rating, justification = parse_response(response)
        except Exception as e:
            print(f"  ERROR on headline {headline_id} (blind): {e}")
            rating, justification = None, str(e)

        results.append({
            "headline_id": headline_id,
            "headline": headline,
            "ground_truth": ground_truth,
            "condition": "blind",
            "source": None,
            "source_tier": None,
            "rating": rating,
            "justification": justification,
            "timestamp": datetime.now().isoformat()
        })

        call_count += 1
        print(f"  [{call_count}/{total_calls}] Headline {headline_id} | Blind | Rating: {rating}")
        time.sleep(SLEEP_BETWEEN_CALLS)

        # ── Condition B: Source-Revealed (one per tier) ───────────────────────
        for tier, source in selected_sources.items():
            source_prompt = source_template.format(headline=headline, source=source)
            try:
                response = call_llm(client, system_prompt, source_prompt)
                rating, justification = parse_response(response)
            except Exception as e:
                print(f"  ERROR on headline {headline_id} (tier {tier}): {e}")
                rating, justification = None, str(e)

            results.append({
                "headline_id": headline_id,
                "headline": headline,
                "ground_truth": ground_truth,
                "condition": "source_revealed",
                "source": source,
                "source_tier": int(tier),
                "rating": rating,
                "justification": justification,
                "timestamp": datetime.now().isoformat()
            })

            call_count += 1
            tier_label = sources_config["tiers"][tier]["label"]
            print(f"  [{call_count}/{total_calls}] Headline {headline_id} | Tier {tier} ({tier_label}): {source} | Rating: {rating}")
            time.sleep(SLEEP_BETWEEN_CALLS)

    # ── Save results ──────────────────────────────────────────────────────────
    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_PATH, index=False)
    print(f"\n✓ Experiment complete. Results saved to {OUTPUT_PATH}")
    print(f"  Total rows: {len(results_df)}")
    print(f"  Parse failures: {results_df['rating'].isna().sum()}")


if __name__ == "__main__":
    run_experiment()
