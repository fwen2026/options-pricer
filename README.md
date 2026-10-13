# options-pricer

Generates implied volatility (IV) surfaces from the Heston model, as synthetic training data for ML volatility-surface research.

## Features

- **Heston pricing:** characteristic function in its numerically stable form, Carr–Madan pricing by FFT (used for surfaces) and by direct quadrature, and a Monte Carlo simulator as a cross-check. FFT and quadrature prices are validated against QuantLib; Monte Carlo is checked against the quadrature price.
- **IV inversion:** vectorized Newton–Raphson with a Brent fallback.
- **Surface generation:** random Heston parameters → FFT price grid → no-arbitrage filtering → IV surface. Settings live in `config/synthetic_iv_datasets/synthetic_iv.yaml`.
- **Stage 1 experiments:** ML approximations of Black–Scholes (NN, XGBoost, Elastic Net). Best was XGBoost at 0.0077 MAE and 2.47% mean relative error; see `notebooks/deprecated/`.

## Future Plans

- Stage 1 (done): ML approximation of Black–Scholes prices
- Stage 2 (done): Heston pricer validated against QuantLib; synthetic IV surface generation
- Stage 3: IV surface learning, using Heston as a fast synthetic data generator
- Stages 4–5: forecast the next-period surface with small specialized models

## Usage

```sh
pip install -r requirements.txt
pytest                                         # pricing and IV solver tests
python scripts/generate_iv_dataset.py          # 10,000 surfaces by default; lower dataset.num_samples for a quick run
python scripts/surface_visualizer.py --surface-id 0
```
