import numpy as np
from src.pricing.black_scholes import BlackScholesOption
from constants import (
    DATA_TRAIN_SIZE, DATA_TEST_SIZE, DATA_EVAL_FRACTION,
    S_RANGE, LOG_MONEYNESS_RANGE, T_RANGE, R_DEFAULT, SIGMA_RANGE, SIGMA_CLIP,
    TRAIN_DATA_PATH, EVAL_DATA_PATH, TEST_DATA_PATH,
    TRAIN_DATA_SEED, TEST_DATA_SEED
)


def generate_train_data(n=DATA_TRAIN_SIZE, seed=TRAIN_DATA_SEED):
    """
    Generate training/eval data in model format.
    X: [log_moneyness, T, r, sigma]
    y: C/K (normalized call price)
    """
    np.random.seed(seed)

    S = np.random.uniform(*S_RANGE, n)
    log_moneyness = np.random.uniform(*LOG_MONEYNESS_RANGE, n)
    K = S / np.exp(log_moneyness)
    T = np.exp(np.random.uniform(np.log(T_RANGE[0]), np.log(T_RANGE[1]), n))
    r = np.full(n, R_DEFAULT)
    sigma = np.clip(np.random.uniform(*SIGMA_RANGE, n), *SIGMA_CLIP)

    option = BlackScholesOption(S, K, T, r, sigma)
    call_prices = option.call_price() / K
    put_prices = option.put_price() / K

    X = np.stack((log_moneyness, T, r, sigma), axis=1).astype(np.float32)
    y_call = np.array(call_prices).astype(np.float32)
    y_put = np.array(put_prices).astype(np.float32)
    return X, y_call, y_put


def generate_test_data(n=DATA_TEST_SIZE, seed=TEST_DATA_SEED):
    """
    Generate test data in raw format for full pipeline testing.
    X: [S, K, T, r, sigma]
    y: C (raw call price)
    """
    np.random.seed(seed)

    S = np.random.uniform(*S_RANGE, n).astype(np.float32)
    log_moneyness = np.random.uniform(*LOG_MONEYNESS_RANGE, n)
    K = (S / np.exp(log_moneyness)).astype(np.float32)
    T = np.exp(np.random.uniform(np.log(T_RANGE[0]), np.log(T_RANGE[1]), n)).astype(np.float32)
    r = np.full(n, R_DEFAULT, dtype=np.float32)
    sigma = np.clip(np.random.uniform(*SIGMA_RANGE, n), *SIGMA_CLIP).astype(np.float32)

    option = BlackScholesOption(S, K, T, r, sigma)
    y_call = option.call_price().astype(np.float32) / K
    y_put = option.put_price().astype(np.float32) / K

    X = np.stack((S, K, T, r, sigma), axis=1)
    return X, y_call, y_put


def partition_train_data(X, y_call, y_put, eval_size=DATA_EVAL_FRACTION, seed=TRAIN_DATA_SEED):
    """
    Partitions training data into train and eval sets.
    """
    np.random.seed(seed)
    n = len(X)
    indices = np.random.permutation(n)
    eval_set_size = int(n * eval_size)
    eval_indices = indices[:eval_set_size]
    train_indices = indices[eval_set_size:]
    return (
        (X[train_indices], y_call[train_indices], y_put[train_indices]),
        (X[eval_indices], y_call[eval_indices], y_put[eval_indices])
    )


def save_datasets(train_filepath=TRAIN_DATA_PATH,
                  eval_filepath=EVAL_DATA_PATH,
                  test_filepath=TEST_DATA_PATH):
    """
    Generate and save all datasets.
    """
    # Training/eval data (model format)
    X, y_call, y_put = generate_train_data()
    (X_train, y_call_train, y_put_train), (X_eval, y_call_eval, y_put_eval) = partition_train_data(X, y_call, y_put)

    np.savez(train_filepath,
             X=X_train,
             y_call=y_call_train,
             y_put=y_put_train,
             n=len(X_train))
    np.savez(eval_filepath,
             X=X_eval,
             y_call=y_call_eval,
             y_put=y_put_eval,
             n=len(X_eval))

    # Test data (raw format)
    X_test, y_call_test, y_put_test = generate_test_data()
    np.savez(test_filepath,
             X=X_test,
             y_call=y_call_test,
             y_put=y_put_test,
             n=len(X_test))

    print(f"Saved train ({len(X_train)}), eval ({len(X_eval)}), test ({len(X_test)}) datasets")


if __name__ == "__main__":
    save_datasets()