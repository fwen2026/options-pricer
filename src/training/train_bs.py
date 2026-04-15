import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.models.bs_model import BSModel

LOG_EPSILON = 1e-8  # small constant to avoid log(0)


def get_device():
    """Get the best available device for training."""
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


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


def compute_normalization_params(X, y):
    """Compute mean and std for inputs and targets."""
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0)
    X_std[X_std < 1e-8] = 1.0  # avoid division by zero for constant features (like r)

    y_log = np.log(y + LOG_EPSILON)
    y_mean = y_log.mean()
    y_std = y_log.std()

    return {
        'X_mean': X_mean,
        'X_std': X_std,
        'y_mean': y_mean,
        'y_std': y_std
    }


def normalize_X(X, params):
    """Standardize input features."""
    return (X - params['X_mean']) / params['X_std']


def normalize_y(y, params):
    """Apply log transform and standardize targets."""
    y_log = np.log(y + LOG_EPSILON)
    return (y_log - params['y_mean']) / params['y_std']


def denormalize_y(y_norm, params):
    """Reverse the normalization to get actual prices."""
    y_log = y_norm * params['y_std'] + params['y_mean']
    return np.exp(y_log) - LOG_EPSILON


def preprocess_data(X, y_call, device, params):
    """
    Preprocess the input data and labels with normalization.
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: The preprocessed input and target tensors.
    """
    X_norm = normalize_X(X, params)
    y_norm = normalize_y(y_call, params)

    X_tensor = torch.tensor(X_norm, dtype=torch.float32).to(device)
    y_call_tensor = torch.tensor(y_norm, dtype=torch.float32).unsqueeze(1).to(device)
    return X_tensor, y_call_tensor


def train_model(model, X_train, y_train, X_eval=None, y_eval=None,
                device=None, epochs=200, lr=1e-3):
    """
    Train the model on provided data.
    Returns the trained model and normalization params.
    """
    if device is None:
        device = get_device()

    # Compute normalization params from training data only
    norm_params = compute_normalization_params(X_train, y_train)

    model.to(device)
    X_tensor, y_tensor = preprocess_data(X_train, y_train, device, norm_params)

    if X_eval is not None and y_eval is not None:
        X_tensor_eval, y_tensor_eval = preprocess_data(X_eval, y_eval, device, norm_params)
    else:
        X_tensor_eval, y_tensor_eval = None, None

    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for n in range(epochs):
        model.train()
        optimizer.zero_grad()

        y_pred = model(X_tensor)
        loss = loss_fn(y_pred, y_tensor)
        loss.backward()
        optimizer.step()

        if n % 10 == 0:
            msg = f"Epoch {n}, Loss: {loss.item():.4f}"
            if X_tensor_eval is not None:
                model.eval()
                with torch.no_grad():
                    y_pred_eval = model(X_tensor_eval)
                    test_loss = loss_fn(y_pred_eval, y_tensor_eval)
                msg += f", Test Loss: {test_loss.item():.4f}"
            print(msg)

    return model, norm_params


def predict(model, X, norm_params, K=None, device=None):
    """
    Run inference and return denormalized prices.
    Use this for all validation and plot generation.

    Args:
        model: The trained model
        X: Input features [moneyness, T, r, sigma]
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


if __name__ == "__main__":
    model = BSModel()
    device = get_device()

    X_train, y_train = load_data('src/data/bs_dataset.npz')
    X_eval, y_eval = load_data('src/data/bs_dataset_eval.npz')

    model, norm_params = train_model(model, X_train, y_train, X_eval, y_eval, device, epochs=200)

    save_model(model, norm_params, 'src/models/bs_model.pth')