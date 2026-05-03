import numpy as np
from src.constants import LOG_EPSILON


def load_data(filepath):
    """
    Load the dataset from a .npz file.
    Returns:
        Tuple[np.ndarray, np.ndarray]: The input features and target labels.
    """
    data = np.load(filepath)
    X = data['X']
    y_call = data['y_call']
    return X, y_call


def compute_normalization_params(X, y, standardize=True):
    """Compute mean and std for inputs and targets."""
    if not standardize:
        return {'standardize': False}

    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0)
    X_std[X_std < 1e-8] = 1.0  # avoid division by zero for constant features (like r)

    y_log = np.log(y + LOG_EPSILON)
    y_mean = y_log.mean()
    y_std = y_log.std()

    return {
        'standardize': True,
        'X_mean': X_mean,
        'X_std': X_std,
        'y_mean': y_mean,
        'y_std': y_std
    }


def normalize_X(X, params):
    """Standardize input features."""
    if not params.get('standardize'):
        return X
    return (X - params['X_mean']) / params['X_std']


def normalize_y(y, params):
    """Apply log transform and standardize targets."""
    if not params.get('standardize'):
        return y
    y_log = np.log(y + LOG_EPSILON)
    return (y_log - params['y_mean']) / params['y_std']


def denormalize_y(y_norm, params):
    """Reverse the normalization to get actual prices."""
    if not params.get('standardize'):
        return y_norm
    y_log = y_norm * params['y_std'] + params['y_mean']
    return np.exp(y_log) - LOG_EPSILON