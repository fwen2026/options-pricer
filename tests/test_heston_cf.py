import sys
sys.path.insert(0, '.')

import time
import numpy as np
from src.pricing.heston.heston_cf import HestonCFModel, CarrMadanSolver

model = HestonCFModel(
    S0=100, v0=0.04,
    theta=0.04, rho=-0.7, kappa=1.5, xi=0.3, r=0.05
)

solver = CarrMadanSolver(model)

# Single price: ATM call, T=1
K = 100
T = 1.0

start = time.perf_counter()
price = solver.solve(K, T)
elapsed = time.perf_counter() - start

print(f"Heston CF call price (K={K}, T={T}): {price:.6f}")
print(f"Time for single price: {elapsed*1000:.2f} ms")

# Sanity checks
print(f"\nSanity checks:")
print(f"  Price > 0: {price > 0}")
print(f"  Price < S0: {price < model.S0}")

# A few more strikes to see the smile
print(f"\nPrices across strikes (T={T}):")
strikes = [80, 90, 95, 100, 105, 110, 120]
times = []
for K in strikes:
    start = time.perf_counter()
    p = solver.solve(K, T)
    t = time.perf_counter() - start
    times.append(t)
    print(f"  K={K:>3}: price={p:.4f}  ({t*1000:.1f} ms)")

print(f"\nAvg time per price: {np.mean(times)*1000:.2f} ms")
print(f"Estimated time for 200-point grid: {np.mean(times)*200:.2f} s")
