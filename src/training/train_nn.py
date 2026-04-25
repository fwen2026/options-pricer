import argparse
import numpy as np
import torch
import torch.optim as optim

from src.models.bs_model import BSModel
from src.training.preprocess_data import (
    load_data, compute_normalization_params, normalize_X, normalize_y, denormalize_y,
)
from src.constants import (
    TRAIN_EPOCHS, TRAIN_LR, TRAIN_PATIENCE,
    TRAIN_MIN_DELTA, TRAIN_LOSS_ALPHA, NN_MODEL_PATH,
    TRAIN_DATA_PATH, EVAL_DATA_PATH, TRAINING_SEED,
)


DEFAULT_HYPERPARAMS = {
    "epochs": TRAIN_EPOCHS,
    "lr": TRAIN_LR,
    "patience": TRAIN_PATIENCE,
    "min_delta": TRAIN_MIN_DELTA,
    "loss_alpha": TRAIN_LOSS_ALPHA,
}


def get_device(use_cpu=False):
    if use_cpu:
        return torch.device("cpu")
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def convert_to_tensor(X_norm, y_norm, device):
    X_tensor = torch.tensor(X_norm, dtype=torch.float32).to(device)
    y_tensor = torch.tensor(y_norm, dtype=torch.float32).unsqueeze(1).to(device)
    return X_tensor, y_tensor


def hybrid_mse_loss(y_pred, y_target, weights, alpha=0.5):
    """
    alpha=0: pure Y-weighted (favors expensive options)
    alpha=1: pure MSE (favors relative accuracy)
    """
    weighted_mse = torch.mean(weights * (y_pred - y_target) ** 2)
    plain_mse = torch.mean((y_pred - y_target) ** 2)
    return (1 - alpha) * weighted_mse + alpha * plain_mse


def fit(model, X_train, y_train, X_eval=None, y_eval=None,
        device=None, epochs=TRAIN_EPOCHS, lr=TRAIN_LR, patience=TRAIN_PATIENCE,
        min_delta=TRAIN_MIN_DELTA, loss_alpha=TRAIN_LOSS_ALPHA,
        y_train_raw=None, y_eval_raw=None, log_every=10):
    """
    Fit model on normalized inputs/targets.
    y_train_raw / y_eval_raw: unnormalized y values used for loss re-weighting.
    """
    if device is None:
        device = get_device()
    model.to(device)

    X_tensor, y_tensor = convert_to_tensor(X_train, y_train, device)
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

        if n % log_every == 0:
            print(msg)

    return model


def train(X_train_raw, y_train_raw, X_eval_raw, y_eval_raw,
          model=None, device=None, **hyperparams):
    """
    End-to-end training from raw arrays. Returns (model, norm_params).
    Extra keyword args (epochs, lr, patience, min_delta, loss_alpha) override
    DEFAULT_HYPERPARAMS. Unknown kwargs raise TypeError.
    """
    hp = {**DEFAULT_HYPERPARAMS, **hyperparams}
    model = model or BSModel()
    device = device or get_device()

    norm_params = compute_normalization_params(X_train_raw, y_train_raw)
    X_train = normalize_X(X_train_raw, norm_params)
    y_train = normalize_y(y_train_raw, norm_params)
    X_eval = normalize_X(X_eval_raw, norm_params)
    y_eval = normalize_y(y_eval_raw, norm_params)

    model = fit(model, X_train, y_train, X_eval, y_eval,
                device=device,
                y_train_raw=y_train_raw, y_eval_raw=y_eval_raw,
                **hp)
    return model, norm_params


def predict(model, X, norm_params, K=None, device=None):
    """
    Run inference on raw 4-feature X; return denormalized prices.
    If K is provided, multiplies C/K back to C.
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
    torch.save({
        'model_state_dict': model.state_dict(),
        'norm_params': norm_params,
    }, filepath)
    print(f"Model saved to {filepath}")


def load_model(filepath, device=None):
    if device is None:
        device = get_device()

    checkpoint = torch.load(filepath, map_location=device, weights_only=False)
    model = BSModel()
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    return model, checkpoint['norm_params']


def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train neural network model")
    parser.add_argument("--epochs", type=int, default=TRAIN_EPOCHS)
    parser.add_argument("--patience", type=int, default=TRAIN_PATIENCE)
    parser.add_argument("--lr", type=float, default=TRAIN_LR)
    args = parser.parse_args()

    set_seed(TRAINING_SEED)
    device = get_device(use_cpu=True)  # CPU for reproducibility

    X_train_raw, y_train_raw = load_data(TRAIN_DATA_PATH)
    X_eval_raw, y_eval_raw = load_data(EVAL_DATA_PATH)

    model, norm_params = train(
        X_train_raw, y_train_raw, X_eval_raw, y_eval_raw,
        device=device,
        epochs=args.epochs, lr=args.lr, patience=args.patience,
    )
    save_model(model, norm_params, NN_MODEL_PATH)
