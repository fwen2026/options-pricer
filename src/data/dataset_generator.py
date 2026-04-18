import numpy as np
from src.pricing.black_scholes import BlackScholesOption


def generate_train_data(n=100000, seed=42):
    """
    Generate training/eval data in model format.
    X: [log_moneyness, T, r, sigma]
    y: C/K (normalized call price)
    """
    np.random.seed(seed)

    S = np.random.uniform(20, 140, n)
    log_moneyness = np.random.uniform(-2, 2, n)
    K = S / np.exp(log_moneyness)
    T = np.exp(np.random.uniform(np.log(0.01), np.log(2.0), n))
    r = np.full(n, 0.3)
    sigma = np.clip(np.random.uniform(0.1, 0.5, n), 0.05, 0.6)

    option = BlackScholesOption(S, K, T, r, sigma)
    call_prices = option.call_price() / K
    put_prices = option.put_price() / K

    X = np.stack((log_moneyness, T, r, sigma), axis=1).astype(np.float32)
    y_call = np.array(call_prices).astype(np.float32)
    y_put = np.array(put_prices).astype(np.float32)
    return X, y_call, y_put


def generate_test_data(n=10000, seed=123):
    """
    Generate test data in raw format for full pipeline testing.
    X: [S, K, T, r, sigma]
    y: C (raw call price)
    """
    np.random.seed(seed)

    S = np.random.uniform(20, 140, n).astype(np.float32)
    log_moneyness = np.random.uniform(-2, 2, n)
    K = (S / np.exp(log_moneyness)).astype(np.float32)
    T = np.exp(np.random.uniform(np.log(0.01), np.log(2.0), n)).astype(np.float32)
    r = np.full(n, 0.3, dtype=np.float32)
    sigma = np.clip(np.random.uniform(0.1, 0.5, n), 0.05, 0.6).astype(np.float32)

    option = BlackScholesOption(S, K, T, r, sigma)
    y_call = option.call_price().astype(np.float32) / K
    y_put = option.put_price().astype(np.float32) / K

    X = np.stack((S, K, T, r, sigma), axis=1)
    return X, y_call, y_put


def partition_train_data(X, y_call, y_put, eval_size=0.1, seed=42):
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


def save_datasets(train_filepath='src/data/bs_dataset.npz',
                  eval_filepath='src/data/bs_dataset_eval.npz',
                  test_filepath='src/data/test_dataset.npz'):
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