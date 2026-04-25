import joblib
import xgboost as xgb

from src.training.preprocess_data import (
    compute_normalization_params, load_data, normalize_y, denormalize_y,
)
from src.constants import EVAL_DATA_PATH, TRAIN_DATA_PATH, XGBOOST_MODEL_PATH


DEFAULT_HYPERPARAMS = {
    "booster": "gbtree",
    "n_estimators": 500,
    "learning_rate": 0.05,
    "max_depth": 6,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_lambda": 1,
    "objective": "reg:squarederror",
    "random_state": 42,
}


def build_model(**overrides):
    """Build an XGBRegressor from DEFAULT_HYPERPARAMS with optional overrides."""
    params = {**DEFAULT_HYPERPARAMS, **overrides}
    return xgb.XGBRegressor(**params)


def fit(model, X_train, y_train, X_eval=None, y_eval=None, verbose=50):
    """Fit an XGBoost model in-place. y should already be normalized."""
    eval_set = [(X_eval, y_eval)] if X_eval is not None and y_eval is not None else None
    model.fit(X_train, y_train, eval_set=eval_set, verbose=verbose)
    return model


def train(X_train, y_train_raw, X_eval, y_eval_raw, **hyperparams):
    """
    End-to-end training from raw (unnormalized) arrays. Returns (model, norm_params).
    """
    norm_params = compute_normalization_params(X_train, y_train_raw)
    y_train = normalize_y(y_train_raw, norm_params)
    y_eval = normalize_y(y_eval_raw, norm_params)

    model = build_model(**hyperparams)
    fit(model, X_train, y_train, X_eval, y_eval)
    return model, norm_params


def predict(model, X, norm_params):
    """Predict and denormalize back to price space."""
    y_raw = model.predict(X)
    return denormalize_y(y_raw, norm_params)


def _norm_params_path(model_path):
    return model_path.replace(".ubj", "_norm_params.joblib")


def save_model(model, norm_params, model_path):
    model.save_model(model_path)
    joblib.dump(norm_params, _norm_params_path(model_path))


def load_model(model_path):
    """Load the XGBoost model and its sidecar normalization parameters."""
    model = xgb.XGBRegressor()
    model.load_model(model_path)
    norm_params = joblib.load(_norm_params_path(model_path))
    return model, norm_params


if __name__ == "__main__":
    X_train, y_train_raw = load_data(TRAIN_DATA_PATH)
    X_eval, y_eval_raw = load_data(EVAL_DATA_PATH)

    model, norm_params = train(X_train, y_train_raw, X_eval, y_eval_raw)
    save_model(model, norm_params, XGBOOST_MODEL_PATH)
