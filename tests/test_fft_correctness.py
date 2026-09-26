import sys
sys.path.insert(0, '.')

import numpy as np
import pytest
import QuantLib as ql

from src.pricing.heston_cf import HestonCFModel, CarrMadanSolver

# Same params as test_heston_correctness.py — Feller: 2*4*0.04 = 0.32 > ξ²=0.09
S0    = 100.0
v0    = 0.04
theta = 0.04
kappa = 4.0
xi    = 0.3
rho   = -0.7
r     = 0.05

STRIKES = [85.0, 100.0, 115.0]

# (n_days, T) pairs where T = n_days/365 exactly.
# Both our solver and QuantLib receive the same T, eliminating day-rounding error
# (int(round(T*365)) introduces up to half a day of skew, which moves ATM prices
# by ~1-2 cents and causes spurious failures at the 1-cent tolerance).
T_CASES = [
    (36,  36  / 365.0),
    (91,  91  / 365.0),
    (182, 182 / 365.0),
    (365, 1.0),
    (730, 2.0),
]

FFT_QUAD_TOL = 0.01   # 1 cent: FFT vs direct quad integration
FFT_QL_TOL   = 0.01   # 1 cent: FFT vs QuantLib analytic engine


def make_solver():
    model = HestonCFModel(S0=S0, v0=v0, theta=theta, rho=rho, kappa=kappa, xi=xi, r=r)
    return CarrMadanSolver(model=model, alpha=1.5, N=4096, eta=0.25)


def nearest_fft_price(K_grid, C_grid, K_target):
    """Return (K_grid_point, price) at the grid point nearest to K_target."""
    idx = np.argmin(np.abs(K_grid - K_target))
    return K_grid[idx], C_grid[idx]


def ql_price(K, n_days):
    today    = ql.Date(1, 1, 2025)
    maturity = today + ql.Period(n_days, ql.Days)
    ql.Settings.instance().evaluationDate = today
    r_ts = ql.YieldTermStructureHandle(ql.FlatForward(today, r, ql.Actual365Fixed()))
    q_ts = ql.YieldTermStructureHandle(ql.FlatForward(today, 0.0, ql.Actual365Fixed()))
    process = ql.HestonProcess(r_ts, q_ts, ql.QuoteHandle(ql.SimpleQuote(S0)),
                                v0, kappa, theta, xi, rho)
    engine  = ql.AnalyticHestonEngine(ql.HestonModel(process))
    option  = ql.VanillaOption(ql.PlainVanillaPayoff(ql.Option.Call, K),
                                ql.EuropeanExercise(maturity))
    option.setPricingEngine(engine)
    return option.NPV()


def test_fft_k_grid_stable_across_expiries():
    """K_grid must be identical for all T.

    generate_single_surface stacks C slices with a single shared K axis
    (it captures K_grid only from the first T slice). If solve_fft ever
    returned a different grid, the strike axis would silently be wrong.
    """
    solver = make_solver()
    T_grid = np.geomspace(0.03, 2.0, 32)
    K_ref, _ = solver.solve_fft(T_grid[0])
    for T in T_grid[1:]:
        K, _ = solver.solve_fft(T)
        np.testing.assert_array_equal(K, K_ref, err_msg=f"K_grid changed at T={T:.4f}")


@pytest.mark.parametrize("n_days,T", T_CASES)
def test_fft_vs_quad(n_days, T):
    """FFT prices agree with direct numerical integration at the same grid strikes."""
    solver = make_solver()
    K_grid, C_grid = solver.solve_fft(T)
    for K_target in STRIKES:
        K_exact, C_fft = nearest_fft_price(K_grid, C_grid, K_target)
        C_quad = solver.solve(K_exact, T)
        assert abs(C_fft - C_quad) < FFT_QUAD_TOL, (
            f"T={T:.5f} ({n_days}d), K≈{K_target} (grid K={K_exact:.4f}): "
            f"FFT={C_fft:.4f}  quad={C_quad:.4f}  diff={abs(C_fft - C_quad):.4f}"
        )


@pytest.mark.parametrize("n_days,T", T_CASES)
def test_fft_vs_quantlib(n_days, T):
    """FFT prices agree with QuantLib's analytic Heston engine at the same grid strikes."""
    solver = make_solver()
    K_grid, C_grid = solver.solve_fft(T)
    for K_target in STRIKES:
        K_exact, C_fft = nearest_fft_price(K_grid, C_grid, K_target)
        C_ql = ql_price(K_exact, n_days)
        assert abs(C_fft - C_ql) < FFT_QL_TOL, (
            f"T={T:.5f} ({n_days}d), K≈{K_target} (grid K={K_exact:.4f}): "
            f"FFT={C_fft:.4f}  QL={C_ql:.4f}  diff={abs(C_fft - C_ql):.4f}"
        )
