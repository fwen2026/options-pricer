import sys
sys.path.insert(0, '.')

import time
import numpy as np
import matplotlib.pyplot as plt
from pricing.heston.heston_mc import HestonModel


# Typical Heston params
model = HestonModel(
    S0=100, v0=0.04,
    theta=0.04, rho=-0.7, kappa=1.5, xi=0.3, r=0.05,
    dt=1e-3
)

T = 1.0
steps = int(T / model.dt)

# --- 1. Visual: 100 paths ---
print("Generating 100 paths for visualization...")
S, v = model.generate_prices(seed=42, steps=steps, paths=100)

fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

time_grid = np.linspace(0, T, steps + 1)

axes[0].plot(time_grid, S, alpha=0.4, linewidth=0.5)
axes[0].axhline(model.S0, color='black', linestyle='--', linewidth=1)
axes[0].set_ylabel("Stock Price")
axes[0].set_title("Heston MC — 100 Sample Paths")

axes[1].plot(time_grid, v, alpha=0.4, linewidth=0.5)
axes[1].axhline(model.theta, color='black', linestyle='--', linewidth=1, label=f"θ={model.theta}")
axes[1].set_ylabel("Variance")
axes[1].set_xlabel("Time (years)")
axes[1].legend()

plt.tight_layout()
plt.savefig("artifacts/plots/heston_mc_paths.png", dpi=150)
plt.close()
print("Saved: artifacts/plots/heston_mc_paths.png")

# --- 2. Martingale test (no-arbitrage) ---
# Under risk-neutral measure, E[S_T * exp(-rT)] should equal S0.
print("\nMartingale test (50k paths)...")
S_large, v_large = model.generate_prices(seed=123, steps=steps, paths=50_000)

S_T = S_large[-1]
discounted_mean = np.exp(-model.r * T) * np.mean(S_T)
martingale_error = abs(discounted_mean - model.S0) / model.S0

print(f"  S0 = {model.S0}")
print(f"  E[S_T * exp(-rT)] = {discounted_mean:.4f}")
print(f"  Relative error = {martingale_error:.6f} ({martingale_error*100:.4f}%)")
if martingale_error < 0.01:
    print("  PASS — martingale property holds (<1% error)")
else:
    print("  FAIL — martingale violation (>1% error), check drift term")

# --- 3. Variance negativity (before clamping) ---
# Re-run without clamping to see how often Euler produces negative variance
print("\nVariance negativity check (50k paths)...")
rng = np.random.default_rng(seed=99)
v_test = np.empty((steps + 1, 50_000))
v_test[0] = model.v0
negative_count = 0
total_count = 0

for t in range(steps):
    dZ1, dZ2 = model.get_random_variables(50_000, rng)
    v_pos = np.maximum(v_test[t], 0)
    v_test[t+1] = v_test[t] + model.get_dv(v_pos, dZ2)
    negative_count += np.sum(v_test[t+1] < 0)
    total_count += 50_000

neg_rate = negative_count / total_count
print(f"  Negative variance occurrences: {negative_count:,} / {total_count:,} ({neg_rate*100:.2f}%)")
print(f"  Feller condition (2*kappa*theta > xi^2): 2*{model.kappa}*{model.theta} = {2*model.kappa*model.theta:.4f} vs xi^2 = {model.xi**2:.4f}")
if 2 * model.kappa * model.theta > model.xi**2:
    print("  Feller condition SATISFIED — negative variance is a discretization artifact")
else:
    print("  Feller condition VIOLATED — negative variance expected even in continuous time")

# --- 4. Timing ---
print("\nTiming benchmarks:")
for n_paths in [1_000, 10_000, 50_000, 100_000]:
    start = time.perf_counter()
    model.generate_prices(seed=0, steps=steps, paths=n_paths)
    elapsed = time.perf_counter() - start
    print(f"  {n_paths:>7,} paths, {steps} steps: {elapsed:.2f}s")
