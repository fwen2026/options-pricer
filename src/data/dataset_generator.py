import numpy as np
from src.pricing.black_scholes import BlackScholesOption

def generate_data(n = 100000, seed = 42):
    """
    Generate synthetic options data.
    """
    
    np.random.seed(seed)
        
    # Generates the data we need.
    S = np.random.uniform(20, 140, n)  # underlying asset prices
    K = np.random.uniform(20, 140, n)  # strike prices
    T = np.random.uniform(0.01, 1, n)  # time to expiration
    r = np.random.uniform(0, 0.1, n)  # risk-free interest rates
    sigma = np.random.uniform(0.1, 0.5, n)  # volatilities
        
    option = BlackScholesOption(S, K, T, r, sigma)

    call_prices = option.call_price()
    put_prices = option.put_price()

    X = np.stack((S, K, T, r, sigma), axis=1).astype(np.float32)
    y_call = np.array(call_prices).astype(np.float32)
    y_put = np.array(put_prices).astype(np.float32)
    return X, y_call, y_put


def partition_data(X, y_call, y_put, test_size=0.2):
    """
    Partitions the data into training and testing sets.
    """
    
    n = len(X)
    indices = np.random.permutation(n)
    test_set_size = int(n * test_size)
    test_indices = indices[:test_set_size]
    train_indices = indices[test_set_size:]
    return (X[train_indices], y_call[train_indices], y_put[train_indices]), (X[test_indices], y_call[test_indices], y_put[test_indices])

def save_dataset(X, y_call, y_put, seed, 
                    train_filepath='src/data/bs_dataset.npz',
                    test_filepath='src/data/bs_dataset_test.npz'):
    """
    Save the training and testing datasets to .npz files.
    """
    
    (X_train, y_call_train, y_put_train), (X_test, y_call_test, y_put_test) = partition_data(X, y_call, y_put)
    
    if y_put is not None:
        np.savez(train_filepath,
                    X=X_train, 
                    y_call=y_call_train, 
                    y_put=y_put_train,
                    seed=seed,
                    n=len(X_train))
        np.savez(test_filepath,
                    X=X_test,
                    y_call=y_call_test,
                    y_put=y_put_test,
                    seed=seed,
                    n=len(X_test))
    else:
        np.savez(train_filepath,
                    X=X_train,
                    y_call=y_call_train,
                    seed=seed,
                    n=len(X_train))
        np.savez(test_filepath,
                    X=X_test,
                    y_call=y_call_test,
                    seed=seed,
                    n=len(X_test))
        

if __name__ == "__main__":
    X, y_call, y_put = generate_data()
    save_dataset(X, y_call, y_put, seed=42)