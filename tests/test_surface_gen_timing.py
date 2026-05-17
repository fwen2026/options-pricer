"""
Time generate_single_surface under the vectorized no-interp pipeline.
Reports finite IV yield and per-phase breakdown to sanity-check FFT params
and IV solver throughput.
"""
import sys
sys.path.insert(0, '.')

import time
import numpy as np
from src.util.synthetic_iv_config import load_synthetic_iv_config, resolve_config
from src.pricing.heston_cf import HestonCFModel, CarrMadanSolver
from data.synthetic_iv_dataset import generate_single_surface, arbitrage_filter


NUM_SURFACES = 10


def time_surface(surface_id, cfg, rng):
    """Run generate_single_surface and collect timing + yield stats."""
    # Snapshot FFT + arb separately so we can attribute BS-invert cost.
    cfg_surface = cfg["surface_grid"]
    rng_snap = np.random.default_rng(rng.integers(2**32))  # reproducible fork

    t_fft = time.perf_counter()
    model = HestonCFModel.from_cfg(cfg, rng=rng_snap)
    solver = CarrMadanSolver(model, cfg=cfg)

    if cfg_surface["log_t"]:
        T_grid = np.geomspace(cfg_surface["t_min"], cfg_surface["t_max"], cfg_surface["t_steps"])
    else:
        T_grid = np.linspace(cfg_surface["t_min"], cfg_surface["t_max"], cfg_surface["t_steps"])
    T_grid = np.sort(T_grid)

    K_grid = None
    price_slices = []
    for T in T_grid:
        K_sl, C_sl = solver.solve_fft(T)
        if K_grid is None:
            K_grid = K_sl
        price_slices.append(C_sl)
    C_grid = np.stack(price_slices, axis=0)
    fft_ms = (time.perf_counter() - t_fft) * 1000

    lm_grid = np.log(model.S0 / K_grid)
    window = (lm_grid >= cfg_surface["lm_min"]) & (lm_grid <= cfg_surface["lm_max"])
    lm_grid = lm_grid[window]
    C_grid = C_grid[:, window]
    sort_idx = np.argsort(lm_grid)
    lm_grid = lm_grid[sort_idx]
    C_grid = C_grid[:, sort_idx]
    K_native = model.S0 / np.exp(lm_grid)

    t_arb = time.perf_counter()
    arb_mask = arbitrage_filter(np.broadcast_to(K_native, C_grid.shape), C_grid)
    arb_ms = (time.perf_counter() - t_arb) * 1000

    arb_violations = int((~arb_mask).sum())
    total_cells = arb_mask.size
    n_lm = lm_grid.size

    # Full timed run via the actual generator (uses the same rng branch)
    t_total = time.perf_counter()
    result = generate_single_surface(surface_id, cfg=cfg, rng=rng)
    total_ms = (time.perf_counter() - t_total) * 1000

    finite_iv = len(result["iv"])
    bs_ms = total_ms - fft_ms - arb_ms  # approximate: remainder is IV inversion

    return {
        "surface_id": surface_id,
        "n_lm": n_lm,
        "total_cells": total_cells,
        "arb_violations": arb_violations,
        "finite_iv": finite_iv,
        "fft_ms": fft_ms,
        "arb_ms": arb_ms,
        "bs_ms": bs_ms,
        "total_ms": total_ms,
    }


def main():
    cfg = load_synthetic_iv_config("config/synthetic_iv_datasets/synthetic_iv.yaml")
    rng = np.random.default_rng(cfg["seed"])

    print(f"FFT N={cfg['fft']['N']}, alpha={cfg['fft']['alpha']}, eta={cfg['fft']['eta']}")
    print(f"T steps: {cfg['surface_grid']['t_steps']}  "
          f"lm window: [{cfg['surface_grid']['lm_min']}, {cfg['surface_grid']['lm_max']}]")
    print(f"Generating {NUM_SURFACES} surfaces (vectorized no-interp)\n")

    header = (
        f"{'id':>3}  {'N_lm':>5}  {'cells':>6}  {'arb':>5}  {'iv':>6}  "
        f"{'fft_ms':>7}  {'arb_ms':>7}  {'bs_ms':>7}  {'total_ms':>8}"
    )
    print(header)
    print("-" * len(header))

    results = []
    for sid in range(NUM_SURFACES):
        r = time_surface(sid, cfg, rng)
        results.append(r)
        print(
            f"{r['surface_id']:>3}  "
            f"{r['n_lm']:>5}  "
            f"{r['total_cells']:>6}  "
            f"{r['arb_violations']:>5}  "
            f"{r['finite_iv']:>6}  "
            f"{r['fft_ms']:>7.1f}  "
            f"{r['arb_ms']:>7.2f}  "
            f"{r['bs_ms']:>7.1f}  "
            f"{r['total_ms']:>8.1f}"
        )

    print("\n=== aggregate ===")
    arb_rates = [r["arb_violations"] / r["total_cells"] for r in results]
    totals_ms = [r["total_ms"] for r in results]
    iv_yields = [r["finite_iv"] / r["total_cells"] for r in results]
    print(f"arb violation rate:  mean={np.mean(arb_rates)*100:.3f}%  max={max(arb_rates)*100:.3f}%")
    print(f"iv yield:            mean={np.mean(iv_yields)*100:.1f}%  min={min(iv_yields)*100:.1f}%")
    print(f"per-surface total:   mean={np.mean(totals_ms):.1f} ms  "
          f"max={max(totals_ms):.1f} ms  min={min(totals_ms):.1f} ms")
    print(f"projected 5000:      {np.mean(totals_ms) * 5000 / 1000:.1f} s")


if __name__ == "__main__":
    main()
