# FIFA World Cup 2026 — Match Predictor & Monte Carlo Simulator

A production-grade pipeline for predicting **FIFA World Cup 2026** match outcomes and simulating the full 48-team tournament tens of thousands of times.

The project combines a classical **Dixon-Coles bivariate Poisson** model for scoreline distributions with an **XGBoost** classifier (Optuna-tuned, isotonic-calibrated) for outcome probabilities, blended into a simple ensemble. The fitted models drive a **vectorised Monte Carlo engine** that runs 50 000 full tournaments end-to-end in under three minutes on a laptop.

---

## Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Python **3.11+** is required.

## How to run

```bash
# 1. Train models + run baseline + scenario simulations
python main.py

# Optional flags
python main.py --skip-training        # reuse cached models in models/saved
python main.py --n-sims 10000         # smaller, faster simulation

# 2. Launch the dashboard
streamlit run dashboard/app.py
```

Outputs:

* `models/saved/` — Dixon-Coles parameters, calibrated XGB pipeline, SHAP & calibration plots.
* `simulation/results/` — `baseline.parquet/csv`, one parquet per scenario, and a `run_meta.json`.

## Model architecture

**Dixon-Coles** decomposes each fixture into independent Poisson draws for home and away goals, with team-specific attack and defence parameters and a low-score correction τ that fixes the well-known under-estimation of 0-0/1-1/0-1/1-0. We fit the parameters by minimising the weighted negative log-likelihood (L-BFGS-B) using exponential time-decay weights tied to the configured half-life so older matches matter less.

**XGBoost** consumes a 20-feature row built from current ELO, log squad value, qualifying xGD, recent form (last-5 win rate), rolling goal averages, head-to-head WC results, and tournament/neutral-ground flags. We pick hyperparameters with Optuna under 5-fold time-series CV, retrain the best model on the full pre-2018 window, and wrap it with `CalibratedClassifierCV` (isotonic). The two heads are then mixed in a convex ensemble (default 0.5/0.5) that the Monte Carlo simulator queries during sampling.

## Key results — top 10 (baseline)

After running `python main.py`, the top probabilities print to stdout and are persisted in `simulation/results/baseline.csv`. Representative numbers from a 50 000-sim baseline run (your numbers will vary by seed and data freshness):

| Team       | Group | P(champion) | P(final) | P(semi) |
|------------|-------|------------:|---------:|--------:|
| Spain      | H     |       17.8% |    28.6% |   42.1% |
| France     | I     |       14.2% |    24.0% |   37.2% |
| England    | L     |       12.9% |    22.7% |   35.8% |
| Argentina  | J     |       10.5% |    19.2% |   30.0% |
| Brazil     | C     |        9.1% |    16.6% |   27.4% |
| Portugal   | K     |        7.4% |    14.3% |   23.6% |
| Germany    | E     |        6.1% |    12.9% |   21.8% |
| Netherlands| F     |        4.0% |     9.6% |   17.2% |
| Norway     | I     |        3.2% |     7.8% |   14.0% |
| Belgium    | G     |        2.5% |     6.3% |   12.1% |

## Limitations

* No live injury / line-up data; player-level effects only enter through the optional scenario overrides.
* Squad values and qualifying xGD are hardcoded snapshots (May 2026) and will go stale.
* The 2026 bracket pathway is approximated — the deterministic seeding template prevents the four highest-rated group winners from meeting before the semifinals, but does not reproduce FIFA's full official re-pairing rules.
* `clubelo.com` does not serve national-team ratings, so the ELO fallback is iteratively computed from match results — fine for relative ordering but not directly comparable to `eloratings.net`.

## Data sources

* Match results: [`martj42/international_results`](https://github.com/martj42/international_results) (CC BY-SA-4.0)
* Penalty shootouts: same repo (`shootouts.csv`)
* Goalscorers: same repo (`goalscorers.csv`)
* ELO API (best-effort): [api.clubelo.com](http://api.clubelo.com)
* Squad values: approximate Transfermarkt snapshots (May 2026, hardcoded)

## Project layout

```
world_cup_sim/
├── data/raw, data/processed   downloaded + engineered datasets
├── models/saved               fitted models + diagnostic plots
├── simulation/results         Monte Carlo outputs (.parquet/.csv)
├── dashboard/app.py           Streamlit UI (6 pages)
├── src/
│   ├── data_loader.py         CSV + ELO + WC 2026 fixtures
│   ├── features.py            per-match feature engineering
│   ├── models.py              Dixon-Coles + XGBoost + ensemble + eval
│   ├── monte_carlo.py         vectorised tournament simulator
│   └── utils.py               config, logging, paths, RNG
├── tests/                     pytest suite
├── main.py                    full pipeline runner
├── config.yaml                tunable parameters
└── requirements.txt
```

## License

MIT
