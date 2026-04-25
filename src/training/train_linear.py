import numpy as np
import joblib
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import GridSearchCV, KFold
from src.training.preprocess_data import load_data, compute_normalization_params, normalize_X, normalize_y, denormalize_y
from src.constants import ELASTIC_MODEL_PATH, TRAIN_DATA_PATH


def engineer_features(X):
    """
    Add polynomial/interaction features for linear model.
    Input: [log_moneyness, T, r, sigma]
    Output: [log_moneyness, T, r, sigma, log_moneyness^2, log(T), sigma*sqrt(T), log_moneyness*sigma]
    """
    log_m, T, r, sigma = X[:, 0], X[:, 1], X[:, 2], X[:, 3]
    return np.column_stack([
        X,
        log_m ** 2,
        np.log(T + 1e-8),
        sigma * np.sqrt(T),
        log_m * sigma
    ])


def train_model(X_train, y_train):
    """
    Training script for a sklearn model. Implements hyperparameter tuning.
    """
    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    model = ElasticNet(max_iter=10000)

    grid = GridSearchCV(estimator=model, param_grid={
        'alpha': [0.1, 1.0, 10.0],
        'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9]
    }, cv=cv)

    grid.fit(X_train, y_train)
    print("Hyperparameters: " + str(grid.best_params_))
    return grid.best_estimator_


def predict(model, X_raw, norm_params, clip_range=(-2, 1.1)):
    """
    Prediction script for a sklearn model
    """
    X_engineered = engineer_features(X_raw)
    X_norm = normalize_X(X_engineered, norm_params)
    y = model.predict(X_norm)
    y = np.clip(y, clip_range[0], clip_range[1])
    return denormalize_y(y, norm_params).flatten()


def load_model(filepath):
    """Load model and normalization params from disk."""
    checkpoint = joblib.load(filepath)
    return checkpoint['model'], checkpoint['norm_params']


if __name__ == "__main__":
    X_raw, y_train = load_data(TRAIN_DATA_PATH)

    # Engineer features on raw X before normalization
    X_engineered = engineer_features(X_raw)

    norm_params = compute_normalization_params(X_engineered, y_train)
    X_norm = normalize_X(X_engineered, norm_params)
    y_norm = normalize_y(y_train, norm_params)

    model = train_model(X_norm, y_norm)

    joblib.dump({
        'model': model,
        'norm_params': norm_params
    }, ELASTIC_MODEL_PATH)