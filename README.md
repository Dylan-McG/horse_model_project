# 🏇 Horse Racing Market Model

An open-source pipeline for probabilistic modelling of UK horse racing, built to measure and exploit market edge, not just position prediction.  
All code is reproducible, and all outputs are programmatically generated for true transparency.

---

## 🚦 Quickstart

git clone https://github.com/yourname/horse_model_project.git
cd horse_model_project
poetry install

# Place raw data in `data/raw/`, market data in `data/market/`

# To run the full pipeline in sequence:
poetry run python run_all.py

# If you prefer pip:
pip install -r requirements.txt
python run_all.py

---

## 📥 Expected Input Data

Place your data as follows:

- `data/raw/train.csv`  
  **Columns:**  
  - `Race_ID` – Unique identifier for each race  
  - `Horse` – Horse name or ID  
  - `Winner` – 1 if this horse won the race, 0 otherwise  
  - `[features...]` – Any additional per-horse features (numeric or categorical)

- `data/raw/test.csv`  
  **Columns:**  
  - Same as `train.csv`, but **without the `Winner` column**

- `data/market/betfair_sp.csv`  
  **Columns:**  
  - `Race_ID`  
  - `Horse`  
  - `Market_Odds` – Market betting odds (e.g. starting price at post time)

> ⚠️ **No Data Leakage:**  
> Only include features that would be available **before the race starts**.  
> Do **not** use results, finishing position, or any post-race info as input features.  
> Market odds (`Market_Odds`) should only be used for market comparison, not as a model input.

> **Note:** Column order doesn’t matter, but headers must match.  
> All files must be CSV, UTF-8 encoded, one row per horse per race.

---

## 📂 Structure

- notebooks/01_eda.ipynb ... 10_market_edge.ipynb  
  Stepwise pipeline from EDA to market exploitation.

- src/  
  Processing and plotting utilities.

- outputs/  
  All model outputs and plots (auto-generated, gitignored).

- data/raw/, data/interim/  
  Your data—never tracked by git.

- pyproject.toml, poetry.lock  
  Fully locked Python environment.

---

## 💡 Highlights

- Stacked Ensemble:  
  LightGBM, RF, XGB, CatBoost, MLP—all blended and stacked.

- Market-Facing:  
  Every output compared to market-implied probabilities, not just win rate.

- Diagnostics:  
  ROI/Sharpe by edge threshold, decile returns, edge histograms, disagreement charts, cumulative P&L.

- Sharpe Optimization:  
  Finds the edge threshold for best risk-adjusted return, with bootstrapped confidence intervals.

- Reproducible:  
  No manual edits—run top-to-bottom and outputs are always fresh.

---

## 📝 For Reviewers

- All outputs are in outputs/—ready for screenshots or reporting.
- See 10_market_edge.ipynb for market exploitation and “alpha.”
- Everything is code-first, so you can audit every step.

---

Run the pipeline, check the outputs, and have fun.

Author: Dylan McGuinness | 2025 | dylanmcguinness0@gmail.com
