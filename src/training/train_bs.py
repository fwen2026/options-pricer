import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.models.bs_model import BSModel

# extract data and convert to tensors
def load_data(filepath):
    data = np.load(filepath)
    X = data['X']
    y_call = data['y_call']
    return X, y_call

def preprocess_data(X, y_call, device):
    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
    y_call_tensor = torch.tensor(y_call, dtype=torch.float32).unsqueeze(1).to(device)
    return X_tensor, y_call_tensor

# model and training device setup
model = BSModel()
device = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

model.to(device)

X_train, y_train = load_data('src/data/bs_dataset.npz')
X_test, y_test = load_data('src/data/bs_dataset_test.npz')
X_tensor, y_call_tensor = preprocess_data(X_train, y_train, device)
X_tensor_test, y_call_tensor_test = preprocess_data(X_test, y_test, device)

loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

epochs = 200

# Start training!
for n in range(epochs):
    model.train()
    optimizer.zero_grad()
    
    y_call_pred = model(X_tensor)
    loss = loss_fn(y_call_pred, y_call_tensor)
    loss.backward()
    optimizer.step()

    model.eval()
    with torch.no_grad():
        y_call_pred_test = model(X_tensor_test)
        test_loss = loss_fn(y_call_pred_test, y_call_tensor_test)

    if n % 10 == 0:
        print(f"Epoch {n}, Loss: {loss.item():.4f}, Test Loss: {test_loss.item():.4f}")