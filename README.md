# IST.700-FinalProject

# Financial News Source Bias in LLMs

A study examining whether revealing the source of a financial news headline shifts an LLM's credibility rating — and by how much.

## Research Question

Does attaching a news source to a financial headline cause an LLM to rate it differently than it would blind? And does that shift correlate with source prestige, revealing inherent LLM bias?

## Project Overview

This project runs a controlled A/B experiment:

- **Condition A (Blind):** LLM rates headline credibility on a 1–5 scale with no source information
- **Condition B (Source-Revealed):** Same headline, same prompt, but with a news source appended

Each headline is tested against sources from four different tiers (High, Mid, Fringe, Fabricated), producing a credibility score delta that quantifies source-driven bias. Results are compared against a logistic regression baseline that uses only headline text.

## Dataset

We use a sample of 100–150 headlines drawn from the **[Fin-Fact dataset](https://github.com/IIT-DM/Fin-Fact/)**, a benchmark dataset for financial fact-checking with labeled credible and misleading entries.

## Source Tiers

| Tier | Sources |
|------|---------|
| **1 – High Prestige** | Wall Street Journal, Bloomberg, Reuters, Financial Times, SEC.gov |
| **2 – Mid-Tier** | Seeking Alpha, Yahoo Finance, MarketWatch, The Motley Fool, ZeroHedge |
| **3 – Fringe** | CoinTelegraph, Natural News, InfoWars |
| **4 – Fabricated** | "National Finance Monitor", "MarketEdge Weekly", "InvestorsTruth.net", "CapitalPulse Daily" |

## Experiment Design

Each headline is evaluated under 5 conditions:
1. Blind (no source)
2. Tier 1 source
3. Tier 2 source
4. Tier 3 source
5. Tier 4 source (fabricated)

This produces **5 ratings per headline × ~125 headlines = ~625 total LLM calls**.

## Repo Structure

```
financial-news-bias/
│
├── README.md
├── requirements.txt
│
├── data/
│   ├── fin_fact_sample.csv       # Sampled headlines with ground truth labels
│   └── sources.json              # Source tier definitions
│
├── prompts/
│   └── prompts.json              # All prompt templates
│
├── scripts/
│   ├── sample_dataset.py         # Sample and prepare Fin-Fact headlines
│   ├── run_experiment.py         # Main experiment runner
│   ├── baseline_model.py         # Logistic regression baseline
│   └── analyze_results.py        # Stats, significance tests, visualizations
│
├── results/
│   ├── raw_outputs.csv           # Every LLM response logged
│   └── summary_stats.csv         # Aggregated results by tier and label
│
├── notebooks/
│   └── analysis.ipynb            # Full analysis walkthrough
│
└── poster/
    └── poster.pdf                # Final project poster
```

## Setup

```bash
git clone https://github.com/YOUR_USERNAME/financial-news-bias.git
cd financial-news-bias
pip install -r requirements.txt
```

Set your OpenAI API key:
```bash
export OPENAI_API_KEY="your-key-here"
```

## Running the Experiment

**Step 1: Prepare the dataset**
```bash
python scripts/sample_dataset.py
```

**Step 2: Run the experiment**
```bash
python scripts/run_experiment.py
```

**Step 3: Run the baseline**
```bash
python scripts/baseline_model.py
```

**Step 4: Analyze results**
```bash
python scripts/analyze_results.py
```
Or open `notebooks/analysis.ipynb` for the full walkthrough.

## Key Metrics

- **Score Delta**: Mean credibility rating change from Blind → Source-Revealed, by tier
- **Bias Rate**: % of misleading headlines rated ≥4 when paired with a Tier 1 source
- **Significance**: Paired t-test / Wilcoxon signed-rank test per tier comparison
- **Baseline Gap**: LLM accuracy vs. logistic regression on credible/misleading classification

## Requirements

See `requirements.txt`. Main dependencies:
- `openai`
- `pandas`
- `scikit-learn`
- `matplotlib`
- `seaborn`
- `scipy`
- `jupyter`

## Citation

If you use the Fin-Fact dataset, please cite the original authors:
> Rangapur et al. (2025). Fin-fact: A benchmark dataset for multimodal financial fact-checking and explanation generation. ACM Web Conference 2025.
