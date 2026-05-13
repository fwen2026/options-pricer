from src.pricing.black_scholes import BlackScholesOption
from src.util.newton_raphson import newton_raphson_solver
from src.util.synthetic_iv_config import resolve_config
from scipy.optimize import brentq
import numpy as np
import math


def implied_vol_solver(S: float, K: float, T: float, r: float, market_price: float,
                       option_type: str = "call",
                       cfg=None) -> float:
    """
    Solve for the implied volatility of a European option using the Black-Scholes model.
    Newton-Raphson with clamped steps, falling back to Brent's on failure.
    """
    option = BlackScholesOption(S=S, K=K, T=T, r=r)
    price_fn = option.call_price if option_type == "call" else option.put_price
    iv_cfg = resolve_config(cfg)["iv_inversion"]
    bounds = (iv_cfg["bounds"]["min"], iv_cfg["bounds"]["max"])
    seed_cfg = iv_cfg["seed"]

    def objective(sigma):
        option.sigma = sigma
        return price_fn() - market_price

    def vega(sigma):
        option.sigma = sigma
        return option.vega()

    def get_seed_sigma():
        log_moneyness = np.log(S / K)
        min_sigma = seed_cfg["min_sigma"]
        atm_threshold = seed_cfg["atm_log_m_threshold"]
        if abs(log_moneyness) < atm_threshold:
            return max(math.sqrt(2 * math.pi / T) * (market_price / S), min_sigma)
        else:
            return max(math.sqrt(2 * abs(log_moneyness + r * T) / T), min_sigma)

    # If Newton Raphson fails, fall back to Brent's method.
    try:
        if iv_cfg["primary_method"] != "newton":
            raise ValueError(f"Unsupported primary IV method: {iv_cfg['primary_method']}")
        implied_vol = newton_raphson_solver(
            objective,
            get_seed_sigma(),
            df=vega,
            tolerance=iv_cfg["newton"]["tolerance"],
            derivative_tolerance=iv_cfg["newton"]["derivative_tolerance"],
            iterations=iv_cfg["newton"]["max_iterations"],
            bounds=bounds,
            numerical_dx=iv_cfg["newton"]["numerical_dx"],
            clamp_to_bounds=iv_cfg["newton"]["clamp_to_bounds"],
        )
    except ValueError:
        if iv_cfg["fallback_method"] != "brentq":
            raise
        implied_vol = brentq(
            objective,
            bounds[0],
            bounds[1],
            xtol=iv_cfg["brentq"]["xtol"],
            maxiter=iv_cfg["brentq"]["max_iterations"],
        )
        
    return implied_vol
