import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

from src.constants import (
    NN_MODEL_PATH, ELASTIC_MODEL_PATH, XGBOOST_MODEL_PATH,
    TRAIN_DATA_PATH, TEST_DATA_PATH, PLOTS_DIR,
)


def preprocess_test_data(X_raw):
    """
    Convert raw test data to model input format.
    X_raw columns: [S, K, T, r, sigma]
    Returns: X_model [log_moneyness, T, r, sigma], K
    """
    S, K, T, r, sigma = X_raw[:, 0], X_raw[:, 1], X_raw[:, 2], X_raw[:, 3], X_raw[:, 4]
    log_moneyness = np.log(S / K)
    X_model = np.stack((log_moneyness, T, r, sigma), axis=1).astype(np.float32)
    return X_model, K


def compute_metrics(y_true, y_pred, min_value_threshold=0.01):
    """
    Compute MAE and relative error metrics.
    Relative errors exclude near-zero values (illiquid options).
    """
    errors = y_pred - y_true
    abs_errors = np.abs(errors)

    mae = np.mean(abs_errors)
    rmse = np.sqrt(np.mean(errors ** 2))
    max_error = np.max(abs_errors)

    mask = np.abs(y_true) >= min_value_threshold
    relative_errors = np.zeros_like(abs_errors)
    relative_errors[mask] = abs_errors[mask] / np.abs(y_true[mask])
    mean_relative_error = np.mean(relative_errors[mask]) if mask.any() else 0.0
    median_relative_error = np.median(relative_errors[mask]) if mask.any() else 0.0

    return {
        'mae': mae,
        'rmse': rmse,
        'mean_relative_error': mean_relative_error,
        'median_relative_error': median_relative_error,
        'max_error': max_error,
        'errors': errors,
        'abs_errors': abs_errors,
        'relative_errors': relative_errors,
    }


def plot_validation_results(X, y_true, y_pred, metrics, style, is_raw_format=True, save_dir=PLOTS_DIR):
    """
    Generate diagnostic plots for model validation.
    is_raw_format: True if X is [S, K, T, r, sigma], False if [log_moneyness, T, r, sigma]
    style: dict with 'color', 'cmap', 'suffix', 'display_name'
    """
    os.makedirs(save_dir, exist_ok=True)

    if is_raw_format:
        S, K, T, sigma = X[:, 0], X[:, 1], X[:, 2], X[:, 4]
        log_moneyness = np.log(S / K)
    else:
        log_moneyness, T, sigma = X[:, 0], X[:, 1], X[:, 3]

    color = style['color']
    _, axes = plt.subplots(2, 3, figsize=(15, 10))

    ax = axes[0, 0]
    ax.scatter(y_true, y_pred, alpha=0.3, s=1, c=color)
    max_val = max(y_true.max(), y_pred.max())
    ax.plot([0, max_val], [0, max_val], 'r--', label='Perfect prediction')
    ax.set_xlabel('Black-Scholes Price')
    ax.set_ylabel('Model Prediction')
    ax.set_title(f'Predicted vs Actual ({style["display_name"]})')
    ax.legend()

    ax = axes[0, 1]
    ax.hist(metrics['errors'], bins=100, alpha=0.7, edgecolor='black', color=color)
    ax.axvline(x=0, color='r', linestyle='--')
    ax.set_xlabel('Error (Predicted - Actual)')
    ax.set_ylabel('Frequency')
    ax.set_title(f'Error Distribution (MAE={metrics["mae"]:.4f})')

    ax = axes[0, 2]
    rel_errors_clipped = np.clip(metrics['relative_errors'], 0, 1)
    ax.hist(rel_errors_clipped, bins=100, alpha=0.7, edgecolor='black', color=color)
    ax.set_xlabel('Relative Error')
    ax.set_ylabel('Frequency')
    ax.set_title(f'Relative Error Dist (Mean={metrics["mean_relative_error"]:.4f})')

    ax = axes[1, 0]
    ax.scatter(log_moneyness, metrics['abs_errors'], alpha=0.3, s=1, c=color)
    ax.set_xlabel('Log Moneyness (log(S/K))')
    ax.set_ylabel('Absolute Error')
    ax.set_title('Error vs Log Moneyness')

    ax = axes[1, 1]
    ax.scatter(T, metrics['abs_errors'], alpha=0.3, s=1, c=color)
    ax.set_xlabel('Time to Expiry (T)')
    ax.set_ylabel('Absolute Error')
    ax.set_title('Error vs Time to Expiry')

    ax = axes[1, 2]
    ax.scatter(sigma, metrics['abs_errors'], alpha=0.3, s=1, c=color)
    ax.set_xlabel('Volatility (sigma)')
    ax.set_ylabel('Absolute Error')
    ax.set_title('Error vs Volatility')

    plt.tight_layout()
    out_path = os.path.join(save_dir, f'validation_results{style["suffix"]}.png')
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved validation plots to {out_path}")

    plot_error_heatmap(log_moneyness, T, metrics['relative_errors'], style, save_dir)


def plot_error_heatmap(log_moneyness, T, relative_errors, style, save_dir):
    """
    Heatmap of relative errors binned by moneyness and time to expiry.
    """
    moneyness_bins = np.linspace(log_moneyness.min(), log_moneyness.max(), 20)
    T_bins = np.linspace(T.min(), T.max(), 20)

    heatmap = np.zeros((len(moneyness_bins) - 1, len(T_bins) - 1))
    counts = np.zeros_like(heatmap)

    m_idx = np.clip(np.digitize(log_moneyness, moneyness_bins) - 1, 0, len(moneyness_bins) - 2)
    t_idx = np.clip(np.digitize(T, T_bins) - 1, 0, len(T_bins) - 2)

    for i in range(len(relative_errors)):
        heatmap[m_idx[i], t_idx[i]] += relative_errors[i]
        counts[m_idx[i], t_idx[i]] += 1

    counts[counts == 0] = 1
    heatmap = heatmap / counts

    _, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(heatmap, aspect='auto', origin='lower',
                   extent=[T_bins.min(), T_bins.max(),
                           moneyness_bins.min(), moneyness_bins.max()],
                   cmap=style['cmap'])
    plt.colorbar(im, label='Mean Relative Error')
    ax.set_xlabel('Time to Expiry (T)')
    ax.set_ylabel('Moneyness (log(S/K))')
    ax.set_title(f'Relative Error Heatmap ({style["display_name"]})')

    plt.tight_layout()
    out_path = os.path.join(save_dir, f'error_heatmap{style["suffix"]}.png')
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved error heatmap to {out_path}")


def print_metrics_report(metrics, display_name):
    print("\n" + "=" * 50)
    print(f"{display_name.upper()} MODEL VALIDATION REPORT")
    print("=" * 50)
    print(f"Mean Absolute Error (MAE):     {metrics['mae']:.6f}")
    print(f"Root Mean Square Error (RMSE): {metrics['rmse']:.6f}")
    print(f"Max Absolute Error:            {metrics['max_error']:.6f}")
    print("-" * 50)
    print(f"Mean Relative Error:           {metrics['mean_relative_error']:.4%}")
    print(f"Median Relative Error:         {metrics['median_relative_error']:.4%}")
    print("=" * 50 + "\n")


def _load_and_predict_nn(model_path):
    from src.training.train_nn import load_model, predict
    model, norm_params = load_model(model_path)
    return lambda X: predict(model, X, norm_params)


def _load_and_predict_linear(model_path):
    from src.training.train_linear import load_model, predict
    model, norm_params = load_model(model_path)
    return lambda X: predict(model, X, norm_params)


def _load_and_predict_xgboost(model_path):
    from src.training.train_xgboost import load_model, predict
    model, norm_params = load_model(model_path)
    return lambda X: predict(model, X, norm_params)


MODEL_CONFIGS = {
    'nn': {
        'default_path': NN_MODEL_PATH,
        'loader': _load_and_predict_nn,
        'style': {'color': 'tab:blue', 'cmap': 'YlOrRd', 'suffix': '', 'display_name': 'NN'},
    },
    'linear': {
        'default_path': ELASTIC_MODEL_PATH,
        'loader': _load_and_predict_linear,
        'style': {'color': 'orange', 'cmap': 'PuBu', 'suffix': '_linear', 'display_name': 'Linear'},
    },
    'xgboost': {
        'default_path': XGBOOST_MODEL_PATH,
        'loader': _load_and_predict_xgboost,
        'style': {'color': 'tab:green', 'cmap': 'YlGn', 'suffix': '_xgboost', 'display_name': 'XGBoost'},
    }
}


DEFAULT_STYLE = {'color': 'tab:gray', 'cmap': 'Greys', 'suffix': '', 'display_name': 'Model'}


def validate_predict_fn(predict_fn, X, y_true, style=None, is_raw_format=True,
                        save_plots=True, save_dir=PLOTS_DIR, display_name=None,
                        report=True):
    """
    Validate any predict_fn against arbitrary (X, y_true) arrays.
    predict_fn takes model-format X [log_m, T, r, sigma] and returns C/K.

    X can be either raw [S, K, T, r, sigma] (is_raw_format=True) or model format.
    Returns metrics dict.
    """
    style = {**DEFAULT_STYLE, **(style or {})}
    if display_name is not None:
        style = {**style, 'display_name': display_name}

    if is_raw_format:
        X_model, _ = preprocess_test_data(X)
        y_pred = predict_fn(X_model)
    else:
        y_pred = predict_fn(X)

    metrics = compute_metrics(y_true, y_pred)
    if report:
        print_metrics_report(metrics, style['display_name'])
    if save_plots:
        plot_validation_results(X, y_true, y_pred, metrics, style,
                                is_raw_format=is_raw_format, save_dir=save_dir)
    return metrics


def validate_model(model_type='nn', model_path=None, use_test_dataset=True):
    """
    File-based validation entry point. Loads model + dataset from disk.
    Tests full pipeline: raw (S, K, T, r, sigma) --> preprocess --> model --> C
    """
    config = MODEL_CONFIGS[model_type]
    model_path = model_path or config['default_path']
    style = config['style']

    predict_fn = config['loader'](model_path)
    print(f"Loaded {style['display_name']} model from {model_path}")

    if use_test_dataset:
        print("\nLoading test dataset...")
        data = np.load(TEST_DATA_PATH)
        X = data['X']
        y_true = data['y_call']
        is_raw = True
    else:
        print("\nLoading training data...")
        data = np.load(TRAIN_DATA_PATH)
        X = data['X']
        y_true = data['y_call']
        is_raw = False

    return validate_predict_fn(predict_fn, X, y_true, style=style, is_raw_format=is_raw)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Validate a model against the analytical BS formula')
    parser.add_argument('--model-type', choices=list(MODEL_CONFIGS.keys()), default='nn',
                        help='Which model family to validate')
    parser.add_argument('--model', type=str, default=None,
                        help='Path to saved model (defaults to the constant for the chosen type)')
    parser.add_argument('--train-data', action='store_true',
                        help='Use training data instead of test dataset')

    args = parser.parse_args()
    validate_model(model_type=args.model_type,
                   model_path=args.model,
                   use_test_dataset=not args.train_data)
