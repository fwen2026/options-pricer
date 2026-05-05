import numpy as np
import joblib
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import GridSearchCV, KFold

from src.training.preprocess_data import (
    load_data, compute_normalization_params, normalize_X, normalize_y, denormalize_y,
)
from src.constants import ELASTIC_MODEL_PATH, TRAIN_DATA_PATH


DEFAULT_PARAM_GRID = {
    'alpha': [0.1, 1.0, 10.0],
    'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9],
}

DEFAULT_CV_SPLITS = 5


def engineer_features(X):
    """
    Add polynomial/interaction features for linear model.
    Input: [log_moneyness, T, r, sigma]
    Output: [log_moneyness, T, r, sigma, log_moneyness^2, log(T), sigma*sqrt(T), log_moneyness*sigma]
    """
    log_m, T, sigma = X[:, 0], X[:, 1], X[:, 3]
    return np.column_stack([
        X,
        log_m ** 2,
        np.log(T + 1e-8),
        sigma * np.sqrt(T),
        log_m * sigma,
    ])


def fit(X_train_norm, y_train_norm, param_grid=None, cv_splits=DEFAULT_CV_SPLITS, seed=42):
    """
    Fit an ElasticNet with grid-searched hyperparameters.
    Inputs are assumed to be engineered + normalized.
    """
    grid = GridSearchCV(
        estimator=ElasticNet(max_iter=10000),
        param_grid=param_grid or DEFAULT_PARAM_GRID,
        cv=KFold(n_splits=cv_splits, shuffle=True, random_state=seed),
    )
    grid.fit(X_train_norm, y_train_norm)
    print("Hyperparameters: " + str(grid.best_params_))
    return grid.best_estimator_


def train(X_train_raw, y_train_raw, param_grid=None, cv_splits=DEFAULT_CV_SPLITS, seed=42):
    """
    End-to-end training from raw arrays. Returns (model, norm_params).
    Engineers features, computes norm params, fits the model.
    """
    X_engineered = engineer_features(X_train_raw)
    norm_params = compute_normalization_params(X_engineered, y_train_raw)
    X_norm = normalize_X(X_engineered, norm_params)
    y_norm = normalize_y(y_train_raw, norm_params)

    model = fit(X_norm, y_norm, param_grid=param_grid, cv_splits=cv_splits, seed=seed)
    return model, norm_params


def predict(model, X_raw, norm_params, clip_range=(-2, 1.1)):
    """Predict from raw 4-feature X and return denormalized prices."""
    X_engineered = engineer_features(X_raw)
    X_norm = normalize_X(X_engineered, norm_params)
    y = model.predict(X_norm)
    y = np.clip(y, clip_range[0], clip_range[1])
    return denormalize_y(y, norm_params).flatten()


def save_model(model, norm_params, filepath):
    joblib.dump({'model': model, 'norm_params': norm_params}, filepath)


def load_model(filepath):
    checkpoint = joblib.load(filepath)
    return checkpoint['model'], checkpoint['norm_params']


if __name__ == "__main__":
    X_raw, y_train_raw = load_data(TRAIN_DATA_PATH)
    model, norm_params = train(X_raw, y_train_raw)
    save_model(model, norm_params, ELASTIC_MODEL_PATH)
