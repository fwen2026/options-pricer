import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.models.bs_model import BSModel


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


def preprocess_data(X, y_call, device):
    """
    Preprocess the input data and labels.
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: The preprocessed input and target tensors.
    """
    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
    y_call_tensor = torch.tensor(y_call, dtype=torch.float32).unsqueeze(1).to(device)
    return X_tensor, y_call_tensor


def train_model(model, X_train, y_train, X_eval=None, y_eval=None,
                device=None, epochs=200, lr=1e-3):
    """
    Train the model on provided data.
    Returns the trained model.
    """
    if device is None:
        device = get_device()

    model.to(device)
    X_tensor, y_tensor = preprocess_data(X_train, y_train, device)

    if X_eval is not None and y_eval is not None:
        X_tensor_eval, y_tensor_eval = preprocess_data(X_eval, y_eval, device)
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

    return model


def save_model(model, filepath):
    """Save model weights to disk."""
    torch.save(model.state_dict(), filepath)
    print(f"Model saved to {filepath}")


if __name__ == "__main__":
    model = BSModel()
    device = get_device()

    X_train, y_train = load_data('src/data/bs_dataset.npz')
    X_eval, y_eval = load_data('src/data/bs_dataset_eval.npz')

    model = train_model(model, X_train, y_train, X_eval, y_eval, device, epochs=200)