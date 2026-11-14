import torch 

W_normal = torch.randn(3, 5, requires_grad=True)

x_normal = torch.randn(5)
y_normal = torch.randn(3)

E = W_normal @ x_normal - y_normal
L = 0.5 * E.pow(2) 

test_gradient = torch.autograd.grad(outputs=L, 
                                    inputs=W_normal, 
                                    grad_outputs=torch.ones_like(L), create_graph=True)[0]

custom_gradient = torch.outer(E.detach(), x_normal)

assert test_gradient.shape == custom_gradient.shape
assert torch.allclose(test_gradient, custom_gradient)
