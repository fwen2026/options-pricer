from dataclasses import dataclass, replace
from typing import Tuple

import numpy as np

from src.pricing.black_scholes import BlackScholesOption
from src.constants import (
    DATA_TRAIN_SIZE, DATA_TEST_SIZE, DATA_EVAL_FRACTION,
    S_RANGE, LOG_MONEYNESS_RANGE, T_RANGE, R_DEFAULT, SIGMA_RANGE, SIGMA_CLIP,
    TRAIN_DATA_PATH, EVAL_DATA_PATH, TEST_DATA_PATH,
    TRAIN_DATA_SEED, TEST_DATA_SEED,
)


@dataclass
class DataConfig:
    """All knobs that define a dataset draw. Defaults come from src.constants."""
    n_train: int = DATA_TRAIN_SIZE
    n_test: int = DATA_TEST_SIZE
    eval_fraction: float = DATA_EVAL_FRACTION

    s_range: Tuple[float, float] = S_RANGE
    log_moneyness_range: Tuple[float, float] = LOG_MONEYNESS_RANGE
    t_range: Tuple[float, float] = T_RANGE
    r: float = R_DEFAULT
    sigma_range: Tuple[float, float] = SIGMA_RANGE
    sigma_clip: Tuple[float, float] = SIGMA_CLIP

    train_seed: int = TRAIN_DATA_SEED
    test_seed: int = TEST_DATA_SEED

    def replace(self, **overrides) -> "DataConfig":
        """Return a copy with the given fields overridden."""
        return replace(self, **overrides)


def generate_train_data(cfg: DataConfig = None):
    """
    Training/eval data in model format.
    X: [log_moneyness, T, r, sigma]
    y: C/K
    """
    cfg = cfg or DataConfig()
    np.random.seed(cfg.train_seed)

    n = cfg.n_train
    S = np.random.uniform(*cfg.s_range, n)
    log_moneyness = np.random.uniform(*cfg.log_moneyness_range, n)
    K = S / np.exp(log_moneyness)
    T = np.exp(np.random.uniform(np.log(cfg.t_range[0]), np.log(cfg.t_range[1]), n))
    r = np.full(n, cfg.r)
    sigma = np.clip(np.random.uniform(*cfg.sigma_range, n), *cfg.sigma_clip)

    option = BlackScholesOption(S, K, T, r, sigma)
    y_call = (option.call_price() / K).astype(np.float32)
    y_put = (option.put_price() / K).astype(np.float32)

    X = np.stack((log_moneyness, T, r, sigma), axis=1).astype(np.float32)
    return X, y_call, y_put


def generate_test_data(cfg: DataConfig = None):
    """
    Test data in raw format for full-pipeline validation.
    X: [S, K, T, r, sigma]
    y: C/K
    """
    cfg = cfg or DataConfig()
    np.random.seed(cfg.test_seed)

    n = cfg.n_test
    S = np.random.uniform(*cfg.s_range, n).astype(np.float32)
    log_moneyness = np.random.uniform(*cfg.log_moneyness_range, n)
    K = (S / np.exp(log_moneyness)).astype(np.float32)
    T = np.exp(np.random.uniform(np.log(cfg.t_range[0]), np.log(cfg.t_range[1]), n)).astype(np.float32)
    r = np.full(n, cfg.r, dtype=np.float32)
    sigma = np.clip(np.random.uniform(*cfg.sigma_range, n), *cfg.sigma_clip).astype(np.float32)

    option = BlackScholesOption(S, K, T, r, sigma)
    y_call = option.call_price().astype(np.float32) / K
    y_put = option.put_price().astype(np.float32) / K

    X = np.stack((S, K, T, r, sigma), axis=1)
    return X, y_call, y_put


def split_train_eval(X, y_call, y_put, eval_fraction: float, seed: int):
    """Split a train-format dataset into train and eval partitions."""
    np.random.seed(seed)
    n = len(X)
    indices = np.random.permutation(n)
    eval_size = int(n * eval_fraction)
    eval_idx = indices[:eval_size]
    train_idx = indices[eval_size:]
    return (
        (X[train_idx], y_call[train_idx], y_put[train_idx]),
        (X[eval_idx], y_call[eval_idx], y_put[eval_idx]),
    )


def generate_train_eval_test(cfg: DataConfig = None):
    """
    Convenience: produce (train, eval, test) triples in one call for notebooks.
    Returns a dict of dicts: {'train': {'X', 'y_call', 'y_put'}, 'eval': {...}, 'test': {...}}
    """
    cfg = cfg or DataConfig()
    X, y_call, y_put = generate_train_data(cfg)
    (X_tr, yc_tr, yp_tr), (X_ev, yc_ev, yp_ev) = split_train_eval(
        X, y_call, y_put, cfg.eval_fraction, cfg.train_seed
    )
    X_te, yc_te, yp_te = generate_test_data(cfg)
    return {
        'train': {'X': X_tr, 'y_call': yc_tr, 'y_put': yp_tr},
        'eval':  {'X': X_ev, 'y_call': yc_ev, 'y_put': yp_ev},
        'test':  {'X': X_te, 'y_call': yc_te, 'y_put': yp_te},
    }


def save_datasets(cfg: DataConfig = None,
                  train_filepath=TRAIN_DATA_PATH,
                  eval_filepath=EVAL_DATA_PATH,
                  test_filepath=TEST_DATA_PATH):
    """Generate under cfg and persist to disk."""
    cfg = cfg or DataConfig()
    bundle = generate_train_eval_test(cfg)

    np.savez(train_filepath,
             X=bundle['train']['X'],
             y_call=bundle['train']['y_call'],
             y_put=bundle['train']['y_put'],
             n=len(bundle['train']['X']))
    np.savez(eval_filepath,
             X=bundle['eval']['X'],
             y_call=bundle['eval']['y_call'],
             y_put=bundle['eval']['y_put'],
             n=len(bundle['eval']['X']))
    np.savez(test_filepath,
             X=bundle['test']['X'],
             y_call=bundle['test']['y_call'],
             y_put=bundle['test']['y_put'],
             n=len(bundle['test']['X']))

    print(f"Saved train ({len(bundle['train']['X'])}), "
          f"eval ({len(bundle['eval']['X'])}), "
          f"test ({len(bundle['test']['X'])}) datasets")


if __name__ == "__main__":
    save_datasets()
