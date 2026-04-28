# IST.700-FinalProject

# Financial News Source Bias in LLMs

A study examining whether revealing the source of a financial news headline shifts an LLM's credibility rating — and by how much.

## Research Question

Does attaching a news source to a financial headline cause an LLM to rate it differently than it would blind? And does that shift correlate with source prestige — revealing inherent LLM bias?

## Project Overview

This project runs a controlled A/B experiment:

- **Condition A (Blind):** LLM rates headline credibility on a 1–5 scale with no source information
- **Condition B (Source-Revealed):** Same headline, same prompt, but with a news source appended

Each headline is tested against sources from four prestige tiers (High, Mid, Fringe, Fabricated), producing a credibility score delta that quantifies source-driven bias. Results are compared against a logistic regression baseline that uses only headline text.

## Dataset

We use a sample of 100 headlines drawn from the **[Fin-Fact dataset](https://github.com/IIT-DM/Fin-Fact/)**, a benchmark dataset for financial fact-checking with labeled credible and misleading entries. Headlines were filtered using financial keywords to ensure domain relevance, then sampled 50/50 between credible and misleading.

## Source Tiers

| Tier | Label | Source Used |
|------|-------|-------------|
| **1** | High Prestige | The Wall Street Journal |
| **2** | Mid-Tier | Yahoo Finance |
| **3** | Fringe | InfoWars |
| **4** | Fabricated | "National Finance Monitor" (fictional) |

## Experiment Design

Each headline is evaluated under 5 conditions:
1. Blind (no source)
2. Tier 1 — Wall Street Journal
3. Tier 2 — Yahoo Finance
4. Tier 3 — InfoWars
5. Tier 4 — National Finance Monitor (fabricated)

This produces **5 ratings per headline × 100 headlines = 500 total LLM calls**.

## Results

### Key Findings

| Condition | Mean Delta vs Blind | p-value | Significant? |
|---|---|---|---|
| Tier 1 — Wall Street Journal | +0.80 | < 0.0001 | ✅ Yes |
| Tier 2 — Yahoo Finance | +0.08 | 0.22 | ❌ No |
| Tier 3 — InfoWars | −1.21 | < 0.0001 | ✅ Yes |
| Tier 4 — National Finance Monitor (fabricated) | −0.11 | 0.10 | ❌ No |

### Prestige Laundering Rate
Percentage of misleading headlines incorrectly rated ≥4 (credible) by condition:

| Condition | Rate |
|---|---|
| Blind (no source) | 14% |
| Tier 1 — Wall Street Journal | 30% |
| Tier 2 — Yahoo Finance | 10% |
| Tier 3 — InfoWars | 0% |
| Tier 4 — Fabricated | 12% |

Attaching the Wall Street Journal name more than doubled the rate at which misleading headlines were mistakenly rated as credible.

### LLM vs Baseline Accuracy

| Model | Accuracy |
|---|---|
| GPT-4o blind condition | 66.0% |
| Logistic Regression + TF-IDF (5-fold CV) | 75.0% |

### Main Conclusion
Source attribution significantly biases LLM credibility ratings. A prestigious source name inflates ratings even for false headlines, while a disreputable source suppresses ratings even for true ones. Fabricated sources with no name recognition had no significant effect — confirming that name recognition, not just source presence, is the mechanism. LLMs should not be used as standalone credibility evaluators when source information is visible.

## Repo Structure

IST.700-FinalProject/
│
├── README.md
├── requirements.txt
│
├── prompts.json              # Exact prompt templates used in experiment
├── sources.json              # Source tier definitions
│
├── run_experiment.py         # Main experiment runner
├── sample_dataset.py         # Sample and prepare Fin-Fact headlines
├── baseline_model.py         # Logistic regression baseline
├── analyze_results.py        # Stats, significance tests, visualizations
│
└── results/
└── raw_outputs.csv       # All 500 LLM responses logged

## Supplemental Materials

- **Google Colab Notebook**: [https://colab.research.google.com/drive/1z0KbD4ODWT0RNSmBx1mB8HTHMIaqmhUU]
- **Raw experiment data**: `raw_outputs.csv`
- **Prompt templates**: `prompts.json`

## Setup

Install dependencies:
```bash
pip install -r requirements.txt
```

Set your OpenAI API key:
```bash
export OPENAI_API_KEY="your-key-here"
```

## Running the Experiment

**Step 1: Prepare the dataset**
```bash
python sample_dataset.py
```

**Step 2: Run the experiment**
```bash
python run_experiment.py
```

**Step 3: Run the baseline**
```bash
python baseline_model.py
```

**Step 4: Analyze results**
```bash
python analyze_results.py
```

## Key Metrics

- **Score Delta**: Mean credibility rating change from Blind → Source-Revealed, by tier
- **Prestige Laundering Rate**: % of misleading headlines rated ≥4 when paired with a Tier 1 source
- **Statistical Significance**: Paired t-test and Wilcoxon signed-rank test per tier comparison
- **Baseline Gap**: LLM accuracy vs. logistic regression on credible/misleading classification

## Requirements

See `requirements.txt`. Main dependencies:
- `openai`
- `pandas`
- `scikit-learn`
- `matplotlib`
- `seaborn`
- `scipy`

## Citation
> Rangapur et al. (2025). Fin-Fact: A benchmark dataset for multimodal financial fact-checking and explanation generation. ACM Web Conference 2025.
