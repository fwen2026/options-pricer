from src.pricing.black_scholes import BlackScholesOption
from src.util.newton_raphson import vectorized_newton_raphson
from src.util.synthetic_iv_config import resolve_config
from scipy.optimize import brentq
import numpy as np
import math


def implied_vol_solver(S_arr: float, K_arr: float, T_arr: float, r_arr: float, market_price_arr: float,
                       option_type: str = "call",
                       cfg=None) -> float:
    """
    Solve for the implied volatility of an array of European options using the Black-Scholes model.
    Newton-Raphson with clamped steps, falling back to Brent's on failure.
    """
    option = BlackScholesOption(S=S_arr, K=K_arr, T=T_arr, r=r_arr)
    price_fn = option.call_price if option_type == "call" else option.put_price
    iv_cfg = resolve_config(cfg)["iv_inversion"]
    bounds = (iv_cfg["bounds"]["min"], iv_cfg["bounds"]["max"])
    seed_cfg = iv_cfg["seed"]

    def objective(sigma_arr):
        option.sigma = sigma_arr
        return price_fn() - market_price_arr

    def vega(sigma_arr):
        option.sigma = sigma_arr
        return option.vega()

    def get_seed_sigma():
        log_moneyness = np.log(S_arr / K_arr)
        min_sigma = seed_cfg["min_sigma"]
        atm_threshold = seed_cfg["atm_log_m_threshold"]

        atm_seed = np.maximum(np.sqrt(2 * math.pi / T_arr) * (market_price_arr / S_arr), min_sigma)
        not_atm_seed = np.maximum(np.sqrt(2 * abs(log_moneyness + r_arr * T_arr) / T_arr), min_sigma)
        
        sigma_seed = np.where(abs(log_moneyness) < atm_threshold, atm_seed, not_atm_seed)
        return np.maximum(sigma_seed, seed_cfg["min_sigma"])

    if iv_cfg["primary_method"] != "newton":
        raise ValueError(f"Unsupported primary IV method: {iv_cfg['primary_method']}")

    # try newton-raphson first. per cell brent's on unconverged elements
    implied_vol_arr, converge_mask = vectorized_newton_raphson(
            objective,
            get_seed_sigma(),
            df=vega,
            tolerance=iv_cfg["newton"]["tolerance"],
            derivative_tolerance=iv_cfg["newton"]["derivative_tolerance"],
            iterations=iv_cfg["newton"]["max_iterations"],
            bounds=bounds,
            numerical_dx=iv_cfg["newton"]["numerical_dx"]
    )

    K_flat = np.ravel(K_arr)
    T_flat = np.ravel(T_arr)
    S_scalar = float(np.ravel(S_arr)[0]) if np.ndim(S_arr) > 0 else float(S_arr)
    r_scalar = float(np.ravel(r_arr)[0]) if np.ndim(r_arr) > 0 else float(r_arr)
    market_price_flat = np.ravel(market_price_arr)

    for i in np.where(~converge_mask)[0]:
        i = int(i)
        cell = BlackScholesOption(S=S_scalar, K=float(K_flat[i]), T=float(T_flat[i]), r=r_scalar)
        cell_price_fn = cell.call_price if option_type == "call" else cell.put_price
        market_i = float(market_price_flat[i])

        def objective_i(sigma, cell=cell, cell_price_fn=cell_price_fn, market_i=market_i):
            cell.sigma = sigma
            return float(cell_price_fn()) - market_i

        implied_vol_arr[i] = brentq(
            objective_i,
            bounds[0],
            bounds[1],
            xtol=iv_cfg["brentq"]["xtol"],
            maxiter=iv_cfg["brentq"]["max_iterations"],
        )
    
    return implied_vol_arr
