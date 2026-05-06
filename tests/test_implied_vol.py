import numpy as np
from src.pricing.black_scholes import BlackScholesOption
from src.pricing.implied_vol_solver import implied_vol_solver


def test_roundtrip_atm_call():
    """Price a call with known sigma, recover sigma via IV solver."""
    S, K, T, r, sigma = 100, 100, 0.5, 0.02, 0.25
    option = BlackScholesOption(S, K, T, r, sigma)
    price = option.call_price()
    iv = implied_vol_solver(S, K, T, r, price, option_type="call")
    assert abs(iv - sigma) < 1e-5, f"Expected {sigma}, got {iv}"


def test_roundtrip_itm_call():
    S, K, T, r, sigma = 120, 100, 1.0, 0.02, 0.30
    option = BlackScholesOption(S, K, T, r, sigma)
    price = option.call_price()
    iv = implied_vol_solver(S, K, T, r, price, option_type="call")
    assert abs(iv - sigma) < 1e-5, f"Expected {sigma}, got {iv}"


def test_roundtrip_otm_call():
    S, K, T, r, sigma = 80, 100, 0.25, 0.02, 0.40
    option = BlackScholesOption(S, K, T, r, sigma)
    price = option.call_price()
    iv = implied_vol_solver(S, K, T, r, price, option_type="call")
    assert abs(iv - sigma) < 1e-5, f"Expected {sigma}, got {iv}"


def test_roundtrip_put():
    S, K, T, r, sigma = 95, 100, 0.5, 0.02, 0.20
    option = BlackScholesOption(S, K, T, r, sigma)
    price = option.put_price()
    iv = implied_vol_solver(S, K, T, r, price, option_type="put")
    assert abs(iv - sigma) < 1e-5, f"Expected {sigma}, got {iv}"


def test_roundtrip_high_vol():
    S, K, T, r, sigma = 100, 100, 0.1, 0.02, 0.80
    option = BlackScholesOption(S, K, T, r, sigma)
    price = option.call_price()
    iv = implied_vol_solver(S, K, T, r, price, option_type="call")
    assert abs(iv - sigma) < 1e-4, f"Expected {sigma}, got {iv}"


def test_roundtrip_low_vol():
    S, K, T, r, sigma = 100, 100, 1.0, 0.02, 0.05
    option = BlackScholesOption(S, K, T, r, sigma)
    price = option.call_price()
    iv = implied_vol_solver(S, K, T, r, price, option_type="call")
    assert abs(iv - sigma) < 1e-5, f"Expected {sigma}, got {iv}"


def test_roundtrip_batch():
    """Sweep across a grid of parameters to catch edge cases."""
    np.random.seed(42)
    failures = []
    for _ in range(100):
        S = np.random.uniform(50, 150)
        K = np.random.uniform(50, 150)
        T = np.random.uniform(0.05, 2.0)
        r = 0.02
        sigma = np.random.uniform(0.05, 0.80)
        option = BlackScholesOption(S, K, T, r, sigma)
        price = option.call_price()
        if price < 1e-10:
            continue
        # Skip options where vega ≈ 0 (IV is numerically unidentifiable)
        option.sigma = sigma
        if option.vega() < 1e-4:
            continue
        try:
            iv = implied_vol_solver(S, K, T, r, float(price), option_type="call")
            if abs(iv - sigma) > 1e-4:
                failures.append((S, K, T, sigma, iv))
        except Exception as e:
            failures.append((S, K, T, sigma, str(e)))
    assert len(failures) == 0, f"{len(failures)} failures: {failures[:5]}"


if __name__ == "__main__":
    test_roundtrip_atm_call()
    test_roundtrip_itm_call()
    test_roundtrip_otm_call()
    test_roundtrip_put()
    test_roundtrip_high_vol()
    test_roundtrip_low_vol()
    test_roundtrip_batch()
    print("All tests passed.")
