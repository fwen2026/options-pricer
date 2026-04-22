import argparse
import numpy as np
import torch
import torch.optim as optim

from src.models.bs_model import BSModel
from src.training.preprocess_data import (
    load_data, compute_normalization_params, normalize_X, normalize_y, denormalize_y
)
from src.constants import (
    TRAIN_EPOCHS, TRAIN_LR, TRAIN_PATIENCE,
    TRAIN_MIN_DELTA, TRAIN_LOSS_ALPHA, MODEL_PATH,
    TRAIN_DATA_PATH, EVAL_DATA_PATH, TRAINING_SEED
)


def get_device(use_cpu=False):
    """Get the best available device for training."""
    if use_cpu:
        return torch.device("cpu")
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def convert_to_tensor(X_norm, y_norm, device):
    """
    Converts preprocessed data as a tensor for training.
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: The preprocessed input and target tensors.
    """
    # X_norm = normalize_X(X, params)
    # y_norm = normalize_y(y_call, params)

    X_tensor = torch.tensor(X_norm, dtype=torch.float32).to(device)
    y_call_tensor = torch.tensor(y_norm, dtype=torch.float32).unsqueeze(1).to(device)
    return X_tensor, y_call_tensor


def hybrid_mse_loss(y_pred, y_target, weights, alpha=0.5):
    """
    Hybrid loss combining Y-weighted MSE (absolute accuracy) and plain MSE (relative accuracy).
    alpha=0: pure Y-weighted (favors expensive options)
    alpha=1: pure MSE (favors relative accuracy)
    """
    weighted_mse = torch.mean(weights * (y_pred - y_target) ** 2)
    plain_mse = torch.mean((y_pred - y_target) ** 2)
    return (1 - alpha) * weighted_mse + alpha * plain_mse


def train_model(model, X_train, y_train, X_eval=None, y_eval=None,
                device=None, epochs=TRAIN_EPOCHS, lr=TRAIN_LR, patience=TRAIN_PATIENCE,
                loss_alpha=TRAIN_LOSS_ALPHA, y_train_raw=None, y_eval_raw=None):
    """
    Train the model on provided data. This data should be preprocessed already.
    y_train_raw/y_eval_raw: original unnormalized y values for computing loss weights.
    """
    if device is None:
        device = get_device()

    model.to(device)
    X_tensor, y_tensor = convert_to_tensor(X_train, y_train, device)

    # Weights based on original y values for Y-weighted component
    y_for_weights = y_train_raw if y_train_raw is not None else y_train
    weights = torch.tensor(y_for_weights, dtype=torch.float32).unsqueeze(1).to(device)
    weights = weights / weights.mean()

    if X_eval is not None and y_eval is not None:
        X_tensor_eval, y_tensor_eval = convert_to_tensor(X_eval, y_eval, device)
        y_eval_for_weights = y_eval_raw if y_eval_raw is not None else y_eval
        weights_eval = torch.tensor(y_eval_for_weights, dtype=torch.float32).unsqueeze(1).to(device)
        weights_eval = weights_eval / weights_eval.mean()
    else:
        X_tensor_eval, y_tensor_eval, weights_eval = None, None, None

    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_eval_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    min_delta = TRAIN_MIN_DELTA

    for n in range(epochs):
        model.train()
        optimizer.zero_grad()

        y_pred = model(X_tensor)
        loss = hybrid_mse_loss(y_pred, y_tensor, weights, loss_alpha)
        loss.backward()
        optimizer.step()
        msg = f"Epoch {n}, Training Loss: {loss.item():.8f}"

        if X_tensor_eval is not None:
            model.eval()
            with torch.no_grad():
                y_pred_eval = model(X_tensor_eval)
                eval_loss = hybrid_mse_loss(y_pred_eval, y_tensor_eval, weights_eval, loss_alpha)
                msg += f", Eval Loss: {eval_loss.item():.8f}"
            eval_loss_item = eval_loss.item()
            if eval_loss_item < best_eval_loss - min_delta:
                best_eval_loss = eval_loss_item
                best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(f"Early stopping at epoch {n}")
                if best_model_state is not None:
                    model.load_state_dict(best_model_state)
                break

        if n % 10 == 0:
            print(msg)

    return model


def predict(model, X, norm_params, K=None, device=None):
    """
    Run inference and return denormalized prices.
    Use this for all validation and plot generation.

    Args:
        model: The trained model
        X: Input features [log moneyness, T, r, sigma]
        norm_params: Normalization parameters from training
        K: Strike prices. If provided, returns actual call price C.
           If None, returns C/K (the normalized price the model outputs).
        device: Torch device
    """
    if device is None:
        device = get_device()

    model.eval()
    X_norm = normalize_X(X, norm_params)
    X_tensor = torch.tensor(X_norm, dtype=torch.float32).to(device)

    with torch.no_grad():
        y_pred_norm = model(X_tensor).cpu().numpy()

    y_pred = denormalize_y(y_pred_norm, norm_params).flatten()

    if K is not None:
        return y_pred * K
    return y_pred


def save_model(model, norm_params, filepath):
    """Save model weights and normalization params to disk."""
    torch.save({
        'model_state_dict': model.state_dict(),
        'norm_params': norm_params
    }, filepath)
    print(f"Model saved to {filepath}")


def load_model(filepath, device=None):
    """Load model weights and normalization params from disk."""
    if device is None:
        device = get_device()

    checkpoint = torch.load(filepath, map_location=device, weights_only=False)
    model = BSModel()
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    norm_params = checkpoint['norm_params']
    return model, norm_params


def set_seed(seed=42):
    """Set seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train neural network model")
    parser.add_argument("--epochs", type=int, default=TRAIN_EPOCHS, help="Number of training epochs")
    parser.add_argument("--patience", type=int, default=TRAIN_PATIENCE, help="Early stopping patience")
    parser.add_argument("--lr", type=float, default=TRAIN_LR, help="Learning rate")
    args = parser.parse_args()

    set_seed(TRAINING_SEED)

    # Use CPU for reproducible training
    device = get_device(use_cpu=True)

    model = BSModel()

    # Preprocess
    X_train, y_train = load_data(TRAIN_DATA_PATH)
    X_eval, y_eval = load_data(EVAL_DATA_PATH)
    norm_params = compute_normalization_params(X_train, y_train)

    X_norm, y_norm = normalize_X(X_train, norm_params), normalize_y(y_train, norm_params)
    X_eval_norm, y_eval_norm = normalize_X(X_eval, norm_params), normalize_y(y_eval, norm_params)
    model = train_model(model, X_norm, y_norm, X_eval_norm, y_eval_norm, device,
                        epochs=args.epochs, lr=args.lr, patience=args.patience,
                        y_train_raw=y_train, y_eval_raw=y_eval)

    save_model(model, norm_params, MODEL_PATH)