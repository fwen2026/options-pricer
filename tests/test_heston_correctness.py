"""
Three-way correctness check for Heston call pricing:
  MC (risk-neutral expectation) vs CF (our Carr-Madan integration) vs QuantLib.

Fixed params with Feller condition satisfied (2κθ > ξ²) to minimise Euler discretisation bias.
"""
import sys
sys.path.insert(0, '.')

import numpy as np
import pytest
import QuantLib as ql

from src.pricing.heston_cf import HestonCFModel, CarrMadanSolver
from src.pricing.heston_mc import HestonMCSim

# Fixed params — Feller: 2*4*0.04 = 0.32 > ξ²=0.09
S0    = 100.0
v0    = 0.04
theta = 0.04
kappa = 4.0
xi    = 0.3
rho   = -0.7
r     = 0.05
T     = 1.0
DT    = 1e-3  # Euler step

STRIKES = [85.0, 100.0, 115.0]
MC_PATHS = 200_000
MC_SEED  = 42

CF_QL_TOL  = 1e-2   # our CF vs QuantLib: should agree to <1 cent
MC_CF_TOL  = 0.75   # MC vs CF: loose — Euler bias + MC variance


def cf_price(K):
    model = HestonCFModel(S0=S0, v0=v0, theta=theta, rho=rho, kappa=kappa, xi=xi, r=r)
    solver = CarrMadanSolver(model=model, alpha=1.5, N=4096, eta=0.25)
    return solver.solve(K, T)


def ql_price(K):
    today    = ql.Date(1, 1, 2025)
    maturity = today + ql.Period(int(T * 365), ql.Days)
    ql.Settings.instance().evaluationDate = today

    r_ts = ql.YieldTermStructureHandle(ql.FlatForward(today, r, ql.Actual365Fixed()))
    q_ts = ql.YieldTermStructureHandle(ql.FlatForward(today, 0.0, ql.Actual365Fixed()))

    process = ql.HestonProcess(r_ts, q_ts, ql.QuoteHandle(ql.SimpleQuote(S0)),
                                v0, kappa, theta, xi, rho)
    engine  = ql.AnalyticHestonEngine(ql.HestonModel(process))

    option = ql.VanillaOption(ql.PlainVanillaPayoff(ql.Option.Call, K),
                               ql.EuropeanExercise(maturity))
    option.setPricingEngine(engine)
    return option.NPV()


def mc_price(K):
    sim   = HestonMCSim(S0=S0, v0=v0, theta=theta, rho=rho, kappa=kappa, xi=xi, r=r, dt=DT)
    steps = int(T / DT)
    S, _  = sim.generate_prices(seed=MC_SEED, steps=steps, paths=MC_PATHS)
    S_T   = S[-1]
    return np.exp(-r * T) * np.mean(np.maximum(S_T - K, 0))


@pytest.mark.parametrize("K", STRIKES)
def test_cf_vs_quantlib(K):
    cf = cf_price(K)
    ql = ql_price(K)
    assert abs(cf - ql) < CF_QL_TOL, (
        f"K={K}: CF={cf:.4f}  QL={ql:.4f}  diff={abs(cf-ql):.4f}"
    )


@pytest.mark.parametrize("K", STRIKES)
def test_mc_vs_cf(K):
    mc = mc_price(K)
    cf = cf_price(K)
    assert abs(mc - cf) < MC_CF_TOL, (
        f"K={K}: MC={mc:.4f}  CF={cf:.4f}  diff={abs(mc-cf):.4f}"
    )


if __name__ == "__main__":
    print(f"Heston params: S0={S0} v0={v0} theta={theta} kappa={kappa} xi={xi} rho={rho} r={r} T={T}")
    feller = 2 * kappa * theta
    print(f"Feller: 2κθ={feller:.3f}  ξ²={xi**2:.3f}  {'SATISFIED' if feller > xi**2 else 'VIOLATED'}\n")
    print(f"{'K':>6}  {'MC':>8}  {'CF':>8}  {'QL':>8}  {'MC-CF':>8}  {'CF-QL':>8}")
    print("-" * 56)
    for K in STRIKES:
        p_mc = mc_price(K)
        p_cf = cf_price(K)
        p_ql = ql_price(K)
        print(f"{K:>6.0f}  {p_mc:>8.4f}  {p_cf:>8.4f}  {p_ql:>8.4f}  "
              f"{p_mc-p_cf:>+8.4f}  {p_cf-p_ql:>+8.4f}")
