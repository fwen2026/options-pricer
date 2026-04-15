"""
Model validation against Black-Scholes analytical formula.
Computes MAE, relative error, and generates diagnostic plots.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import torch
import matplotlib.pyplot as plt

from src.models.bs_model import BSModel
from src.pricing.black_scholes import BlackScholesOption
from src.training.train_bs import load_data, preprocess_data, train_model, get_device


def load_model(model_path=None, device=None):
    """
    Load a trained model from disk, or train a new one if no path provided.
    """
    if device is None:
        device = get_device()

    model = BSModel()
    model.to(device)

    if model_path and os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        print(f"Loaded model from {model_path}")
    else:
        print("No saved model found. Training a new model...")
        X_train, y_train = load_data('src/data/bs_dataset.npz')
        X_test, y_test = load_data('src/data/bs_dataset_test.npz')
        model = train_model(model, X_train, y_train, X_test, y_test, device, epochs=200)

    model.eval()
    return model, device


def compute_black_scholes_prices(X):
    """
    Compute analytical Black-Scholes call prices for input data.
    X columns: [S, K, T, r, sigma]
    """
    S, K, T, r, sigma = X[:, 0], X[:, 1], X[:, 2], X[:, 3], X[:, 4]
    option = BlackScholesOption(S, K, T, r, sigma)
    return option.call_price()


def compute_metrics(y_true, y_pred):
    """
    Compute MAE and relative error metrics.
    """
    errors = y_pred - y_true
    abs_errors = np.abs(errors)

    mae = np.mean(abs_errors)

    epsilon = 1e-8
    relative_errors = abs_errors / (np.abs(y_true) + epsilon)
    mean_relative_error = np.mean(relative_errors)
    median_relative_error = np.median(relative_errors)

    rmse = np.sqrt(np.mean(errors ** 2))
    max_error = np.max(abs_errors)

    return {
        'mae': mae,
        'rmse': rmse,
        'mean_relative_error': mean_relative_error,
        'median_relative_error': median_relative_error,
        'max_error': max_error,
        'errors': errors,
        'abs_errors': abs_errors,
        'relative_errors': relative_errors
    }


def plot_validation_results(X, y_true, y_pred, metrics, save_dir='test/plots'):
    """
    Generate diagnostic plots for model validation.
    """
    os.makedirs(save_dir, exist_ok=True)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    ax = axes[0, 0]
    ax.scatter(y_true, y_pred, alpha=0.3, s=1)
    max_val = max(y_true.max(), y_pred.max())
    ax.plot([0, max_val], [0, max_val], 'r--', label='Perfect prediction')
    ax.set_xlabel('Black-Scholes Price')
    ax.set_ylabel('Model Prediction')
    ax.set_title('Predicted vs Actual')
    ax.legend()

    ax = axes[0, 1]
    ax.hist(metrics['errors'], bins=100, alpha=0.7, edgecolor='black')
    ax.axvline(x=0, color='r', linestyle='--')
    ax.set_xlabel('Error (Predicted - Actual)')
    ax.set_ylabel('Frequency')
    ax.set_title(f'Error Distribution (MAE={metrics["mae"]:.4f})')

    ax = axes[0, 2]
    rel_errors_clipped = np.clip(metrics['relative_errors'], 0, 1)
    ax.hist(rel_errors_clipped, bins=100, alpha=0.7, edgecolor='black')
    ax.set_xlabel('Relative Error')
    ax.set_ylabel('Frequency')
    ax.set_title(f'Relative Error Dist (Mean={metrics["mean_relative_error"]:.4f})')

    ax = axes[1, 0]
    moneyness = np.log(X[:, 0] / X[:, 1])
    ax.scatter(moneyness, metrics['abs_errors'], alpha=0.3, s=1)
    ax.set_xlabel('Moneyness (log(S/K))')
    ax.set_ylabel('Absolute Error')
    ax.set_title('Error vs Moneyness')

    ax = axes[1, 1]
    T = X[:, 2]
    ax.scatter(T, metrics['abs_errors'], alpha=0.3, s=1)
    ax.set_xlabel('Time to Expiry (T)')
    ax.set_ylabel('Absolute Error')
    ax.set_title('Error vs Time to Expiry')

    ax = axes[1, 2]
    sigma = X[:, 4]
    ax.scatter(sigma, metrics['abs_errors'], alpha=0.3, s=1)
    ax.set_xlabel('Volatility (sigma)')
    ax.set_ylabel('Absolute Error')
    ax.set_title('Error vs Volatility')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'validation_results.png'), dpi=150)
    plt.close()
    print(f"Saved validation plots to {save_dir}/validation_results.png")

    plot_error_heatmap(X, metrics['relative_errors'], save_dir)


def plot_error_heatmap(X, relative_errors, save_dir):
    """
    Create a heatmap of relative errors binned by moneyness and time to expiry.
    """
    moneyness = np.log(X[:, 0] / X[:, 1])
    T = X[:, 2]

    moneyness_bins = np.linspace(moneyness.min(), moneyness.max(), 20)
    T_bins = np.linspace(T.min(), T.max(), 20)

    heatmap = np.zeros((len(moneyness_bins) - 1, len(T_bins) - 1))
    counts = np.zeros_like(heatmap)

    m_idx = np.clip(np.digitize(moneyness, moneyness_bins) - 1, 0, len(moneyness_bins) - 2)
    t_idx = np.clip(np.digitize(T, T_bins) - 1, 0, len(T_bins) - 2)

    for i in range(len(relative_errors)):
        heatmap[m_idx[i], t_idx[i]] += relative_errors[i]
        counts[m_idx[i], t_idx[i]] += 1

    counts[counts == 0] = 1
    heatmap = heatmap / counts

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(heatmap, aspect='auto', origin='lower',
                   extent=[T_bins.min(), T_bins.max(),
                          moneyness_bins.min(), moneyness_bins.max()],
                   cmap='YlOrRd')
    plt.colorbar(im, label='Mean Relative Error')
    ax.set_xlabel('Time to Expiry (T)')
    ax.set_ylabel('Moneyness (log(S/K))')
    ax.set_title('Relative Error Heatmap')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'error_heatmap.png'), dpi=150)
    plt.close()
    print(f"Saved error heatmap to {save_dir}/error_heatmap.png")


def print_metrics_report(metrics):
    """
    Print a formatted metrics report.
    """
    print("\n" + "=" * 50)
    print("MODEL VALIDATION REPORT")
    print("=" * 50)
    print(f"Mean Absolute Error (MAE):     {metrics['mae']:.6f}")
    print(f"Root Mean Square Error (RMSE): {metrics['rmse']:.6f}")
    print(f"Max Absolute Error:            {metrics['max_error']:.6f}")
    print("-" * 50)
    print(f"Mean Relative Error:           {metrics['mean_relative_error']:.4%}")
    print(f"Median Relative Error:         {metrics['median_relative_error']:.4%}")
    print("=" * 50 + "\n")


def validate_model(model_path=None, use_test_dataset=True):
    """
    Main validation function.
    """
    model, device = load_model(model_path)

    if use_test_dataset:
        print("\nLoading test dataset...")
        X_test, y_true = load_data('src/data/bs_dataset_test.npz')
    else:
        print("\nComputing Black-Scholes prices from training data...")
        X_test, _ = load_data('src/data/bs_dataset.npz')
        y_true = compute_black_scholes_prices(X_test)

    X_tensor, _ = preprocess_data(X_test, y_true, device)
    with torch.no_grad():
        y_pred = model(X_tensor).cpu().numpy().flatten()

    metrics = compute_metrics(y_true, y_pred)
    print_metrics_report(metrics)
    plot_validation_results(X_test, y_true, y_pred, metrics)

    return metrics


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Validate BS model against analytical formula')
    parser.add_argument('--model', type=str, default=None,
                        help='Path to saved model (.pth file)')
    parser.add_argument('--train-data', action='store_true',
                        help='Use training data instead of test dataset')

    args = parser.parse_args()
    validate_model(model_path=args.model, use_test_dataset=not args.train_data)
