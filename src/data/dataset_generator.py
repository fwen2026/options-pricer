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

    feature_mode: str = "log-moneyness"

    def replace(self, **overrides) -> "DataConfig":
        """Return a copy with the given fields overridden."""
        return replace(self, **overrides)
    
@dataclass
class Dataset:
    """ A dataset object for option pricing data. """
    X: np.ndarray
    y_call: np.ndarray
    y_put: np.ndarray
    
@dataclass
class DatasetBundle:
    """ A bundle of train/val/test datasets. """
    train: Dataset
    val: Dataset
    test: Dataset


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
    if cfg.feature_mode == "log-moneyness":
        X = np.stack((log_moneyness, T, r, sigma), axis=1).astype(np.float32)
    elif cfg.feature_mode == "raw":
        X = np.stack((S, K, T, r, sigma), axis=1).astype(np.float32)
    else:
        raise ValueError(f"Unknown feature mode: {cfg.feature_mode}")
    return Dataset(X, y_call, y_put)


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
    return Dataset(X, y_call, y_put)


def split_train_val(dataset: Dataset, val_fraction: float, seed: int):
    """Split a train-format dataset into (train, val) partitions."""
    np.random.seed(seed)
    n = len(dataset.X)
    indices = np.random.permutation(n)
    val_size = int(n * val_fraction)
    val_idx = indices[:val_size]
    train_idx = indices[val_size:]
    return (
        Dataset(dataset.X[train_idx], dataset.y_call[train_idx], dataset.y_put[train_idx]),
        Dataset(dataset.X[val_idx], dataset.y_call[val_idx], dataset.y_put[val_idx]),
    )


def generate_bundle(cfg: DataConfig = None) -> DatasetBundle:
    """Produce a train/val/test bundle in one call."""
    cfg = cfg or DataConfig()
    full_train = generate_train_data(cfg)
    train, val = split_train_val(full_train, cfg.eval_fraction, cfg.train_seed)
    test = generate_test_data(cfg)
    return DatasetBundle(train=train, val=val, test=test)


def _save_dataset(dataset: Dataset, filepath: str):
    np.savez(filepath,
             X=dataset.X,
             y_call=dataset.y_call,
             y_put=dataset.y_put,
             n=len(dataset.X))


def save_datasets(cfg: DataConfig = None,
                  train_filepath=TRAIN_DATA_PATH,
                  eval_filepath=EVAL_DATA_PATH,
                  test_filepath=TEST_DATA_PATH):
    """Generate under cfg and persist to disk."""
    cfg = cfg or DataConfig()
    bundle = generate_bundle(cfg)

    _save_dataset(bundle.train, train_filepath)
    _save_dataset(bundle.val, eval_filepath)
    _save_dataset(bundle.test, test_filepath)

    print(f"Saved train ({len(bundle.train.X)}), "
          f"val ({len(bundle.val.X)}), "
          f"test ({len(bundle.test.X)}) datasets")


if __name__ == "__main__":
    save_datasets()
