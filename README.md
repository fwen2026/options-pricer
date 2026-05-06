# Options Pricer

A machine learning research project comparing how well different models approximate the Black-Scholes option pricing formula. Trained on synthetic data generated from the analytical BS solution, with evaluation across moneyness and expiry regimes.

## Models

- **Neural Network** (PyTorch) — 4-layer MLP with configurable architecture, hybrid MSE loss, log-space target transform
- **XGBoost** — gradient boosted trees with the same feature engineering
- **Elastic Net** — linear baseline with polynomial features

## Key Results

| Model | MAE | RMSE | Max Error | Mean Rel. Error |
|-------|-----|------|-----------|-----------------|
| XGB (no standardization) | **0.0077** | **0.0124** | **0.068** | **2.47%** |
| NN (no standardization) | 0.0165 | 0.0261 | 0.224 | 8.69% |
| XGB baseline | 0.0308 | 0.0754 | 1.075 | 4.24% |
| NN baseline | 0.0478 | 0.1075 | 1.897 | 8.39% |

Non-standardized XGBoost achieves near-perfect approximation (MAE < 0.01) on this problem.

## Findings

- **Standardization hurts on clean analytic data.** Log-transform + z-score warps the target into a harder optimization landscape without benefit.
- **Feature engineering (log-moneyness) is significant.** Both models degrade 3-5x without it. The network must otherwise learn that only S/K matters.
- **Hybrid loss improves tail accuracy** at the expense of aggregate metrics. The y-weighted MSE term pulls gradient toward cheap OTM options.
- **NN architecture affects error magnitude, not shape.** The residual error pattern is structural to the MLP, not a capacity issue.

## Structure

```
src/
  pricing/        Black-Scholes pricer, implied vol solver
  models/         PyTorch NN architecture
  training/       Training scripts (NN, XGBoost, Elastic Net)
  eval/           Validation and plotting
  data/           Synthetic dataset generation
  util/           Newton-Raphson, bisection solvers
notebooks/        Ablation studies and performance comparisons
artifacts/        Saved models, datasets, plots
```

## Usage

```bash
python -m src.data.dataset_generator

python -m src.training.train_nn

python -m src.training.train_xgboost

python -m src.eval.validate --model-type xgboost
```

## Requirements

```
numpy, scipy, torch, matplotlib, scikit-learn, xgboost, pandas
```
Also see `requirements.txt`
