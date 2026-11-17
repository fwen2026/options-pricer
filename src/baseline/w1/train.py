import torch
import torch.optim.adam as adam
import numpy as np
from src.baseline.w1.pipeline import tvt_split, load_toy_block
from src.baseline.w1.mlp import create_mlp


def mask_batch(batch, threshold=0.25):
    """Masks certain values of surfaces for a dataset. Also appends the mask to each surface"""
    B = batch.shape
    
    masked_b = torch.rand(B) >= threshold
    mask = masked_b.float()

    return torch.cat([batch.masked_fill(masked_b, 0), mask], dim=1), masked_b


def loss_fn(pred, target, mask):
    """MSE loss of masked values"""
    mean_error_tensor = (pred - target) ** 2
    masked_mean_e = mean_error_tensor.masked_fill(~mask, -1)

    mse = masked_mean_e[masked_mean_e >= 0].mean()
    return mse

def train_loop(X, model, opt, val_X, epochs=100):
    idx = X.shape[0]
    val_in, val_hidden = mask_batch(val_X)
    for epoch in range(epochs):
        ord = torch.randperm(idx)
        for i in range(0, idx, 128):
            batch_X = X[ord[i:i+128]]
            x, hidden = mask_batch(batch_X)

            pred = model(x)
            loss = loss_fn(pred, batch_X, hidden)
            
            opt.zero_grad() # zeroes the gradients 
            loss.backward() # finds the gradients for each layer and each weight
            opt.step() # moves each weight backwards against the gradient

        with torch.no_grad():
            val_loss = loss_fn(model(val_in), val_X, val_hidden)
            print(f"Epoch {epoch}: {val_loss.item()}")

    return np.sqrt(val_loss.item())


if __name__ == "__main__":
    iv, T, log_m = load_toy_block() 
    train, val, test = tvt_split(iv)

    train = torch.from_numpy(train)
    val = torch.from_numpy(val)
    test = torch.from_numpy(test)

    train, val = train.flatten(start_dim=1), val.flatten(start_dim=1)
    
    # test for loss fn
    pred = torch.tensor([
        [1., 2., 3.],
        [4., 5., 6.]
    ])

    act = torch.tensor([
        [1., 0., 3.],
        [0., 5., 6.]
    ])
    
    mask = torch.tensor([
        [False, True, False],
        [True, False, False]
    ])

    assert np.allclose(loss_fn(pred, act, mask), 10)
    
    model = create_mlp(660, 330, [660, 660])
    opt = adam.Adam(model.parameters(), 1e-3) #1e-3 is the step per grad.
    
    rmse = train_loop(train, model, opt, val)
    print(f"Final RMSE {rmse}")

    # tst = train[:8]
    # tst_x, hidden = mask_batch(tst)
    
    # for s in range(1000):
    #     pred = model(tst_x)
    #     loss = loss_fn(pred, tst, hidden)
        
    #     opt.zero_grad() # zeroes the gradients 
    #     loss.backward() # finds the gradients for each layer and each weight
    #     opt.step() # moves each weight backwards against the gradient

    #     if s % 100 == 0:
    #         print(f"Step {s}: {loss.item()}")
