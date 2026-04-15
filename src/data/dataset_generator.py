import numpy as np
from src.pricing.black_scholes import BlackScholesOption

def generate_data(n = 100000, seed = 42):
    """
    Generate synthetic options data.
    """
    
    np.random.seed(seed)
        
    # Generates the data we need.
    S = np.random.uniform(20, 140, n)  # underlying asset prices
    moneyness = np.random.normal(1.0, 0.2, n)
    K = S * moneyness # strike prices
    T = np.random.uniform(0.01, 1, n)  # time to expiration
    r = np.full(n, 0.3)  # risk-free interest rates
    sigma = np.clip(np.random.uniform(0.1, 0.5, n), 0.05, 0.6)  # volatilities

    option = BlackScholesOption(S, K, T, r, sigma)

    call_prices = option.call_price() / K
    put_prices = option.put_price() / K

    X = np.stack((moneyness, T, r, sigma), axis=1).astype(np.float32) # train the model ONLY on moneyness. take care of denormalization ourselves.
    y_call = np.array(call_prices).astype(np.float32)
    y_put = np.array(put_prices).astype(np.float32)
    K = K.astype(np.float32)
    return X, y_call, y_put, K


def partition_data(X, y_call, y_put, K, eval_size=0.1, test_size=0.1):
    """
    Partitions the data into training, eval, and test sets.
    """

    n = len(X)
    indices = np.random.permutation(n)
    eval_set_size = int(n * eval_size)
    test_set_size = int(n * test_size)
    eval_indices = indices[:eval_set_size]
    test_indices = indices[eval_set_size:eval_set_size + test_set_size]
    train_indices = indices[eval_set_size + test_set_size:]
    return (
        (X[train_indices], y_call[train_indices], y_put[train_indices], K[train_indices]),
        (X[eval_indices], y_call[eval_indices], y_put[eval_indices], K[eval_indices]),
        (X[test_indices], y_call[test_indices], y_put[test_indices], K[test_indices])
    )

def save_dataset(X, y_call, y_put, K, seed,
                    train_filepath='src/data/bs_dataset.npz',
                    eval_filepath='src/data/bs_dataset_eval.npz',
                    test_filepath='src/data/test_dataset.npz'):
    """
    Save the training, evaluation, and test datasets to .npz files.
    """

    (X_train, y_call_train, y_put_train, K_train), (X_eval, y_call_eval, y_put_eval, K_eval), (X_test, y_call_test, y_put_test, K_test) = partition_data(X, y_call, y_put, K)

    if y_put is not None:
        np.savez(train_filepath,
                    X=X_train,
                    y_call=y_call_train,
                    y_put=y_put_train,
                    K=K_train,
                    seed=seed,
                    n=len(X_train))
        np.savez(eval_filepath,
                    X=X_eval,
                    y_call=y_call_eval,
                    y_put=y_put_eval,
                    K=K_eval,
                    seed=seed,
                    n=len(X_eval))
        np.savez(test_filepath,
                    X=X_test,
                    y_call=y_call_test,
                    y_put=y_put_test,
                    K=K_test,
                    seed=seed,
                    n=len(X_test))
    else:
        np.savez(train_filepath,
                    X=X_train,
                    y_call=y_call_train,
                    K=K_train,
                    seed=seed,
                    n=len(X_train))
        np.savez(eval_filepath,
                    X=X_eval,
                    y_call=y_call_eval,
                    K=K_eval,
                    seed=seed,
                    n=len(X_eval))
        np.savez(test_filepath,
                    X=X_test,
                    y_call=y_call_test,
                    K=K_test,
                    seed=seed,
                    n=len(X_test))


if __name__ == "__main__":
    X, y_call, y_put, K = generate_data()
    save_dataset(X, y_call, y_put, K, seed=42)