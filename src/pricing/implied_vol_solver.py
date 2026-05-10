from src.pricing.black_scholes import BlackScholesOption
from src.util.newton_raphson import newton_raphson_solver
from scipy.optimize import brentq
import numpy as np
import math


def implied_vol_solver(S: float, K: float, T: float, r: float, market_price: float,
                       option_type: str = "call") -> float:
    """
    Solve for the implied volatility of a European option using the Black-Scholes model.
    Newton-Raphson with clamped steps, falling back to Brent's on failure.
    """
    option = BlackScholesOption(S=S, K=K, T=T, r=r)
    price_fn = option.call_price if option_type == "call" else option.put_price

    def objective(sigma):
        option.sigma = sigma
        return price_fn() - market_price

    def vega(sigma):
        option.sigma = sigma
        return option.vega()

    def get_seed_sigma():
        log_moneyness = np.log(S / K)
        if abs(log_moneyness) < 0.5:
            return max(math.sqrt(2 * math.pi / T) * (market_price / S), 0.01)
        else:
            return max(math.sqrt(2 * abs(log_moneyness + r * T) / T), 0.01)

    # If Newton Raphson fails, fall back to Brent's method.
    try:
        implied_vol = newton_raphson_solver(objective, get_seed_sigma(), df=vega)
    except ValueError:
        implied_vol = brentq(objective, 1e-6, 5.0, xtol=1e-8, maxiter=200)
        
    return implied_vol
