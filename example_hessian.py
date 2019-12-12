import torch
from torch import nn

from autograd_lib import autograd_lib
from collections import defaultdict

from attrdict import AttrDefault

def simple_model(d, num_layers):
    """Creates simple linear neural network initialized to identity"""
    layers = []
    for i in range(num_layers):
        layer = nn.Linear(d, d, bias=False)
        layer.weight.data.copy_(torch.eye(d))
        layers.append(layer)
    return torch.nn.Sequential(*layers)


def least_squares(data, targets=None):
    """Least squares loss (like MSELoss, but an extra 1/2 factor."""
    if targets is None:
        targets = torch.zeros_like(data)
    err = data - targets.view(-1, data.shape[1])
    return torch.sum(err * err) / 2 / len(data)

d=1
n=1
model = simple_model(1, 5)
data = torch.ones((n, d))
targets = torch.ones((n, d))
loss_fn = least_squares

autograd_lib.register(model)

hess = defaultdict(float)
hess_diag = defaultdict(float)
hess_kfac = defaultdict(lambda: AttrDefault(float))

activations = {}
def save_activations(layer, A, _):
    activations[layer] = A

    # KFAC left factor
    hess_kfac[layer].AA += torch.einsum("ni,nj->ij", A, A)

with autograd_lib.module_hook(save_activations):
    output = model(data)
    loss = loss_fn(output, targets)

def compute_hess(layer, _, B):
    A = activations[layer]
    BA = torch.einsum("nl,ni->nli", B, A)

    # full Hessian
    hess[layer] += torch.einsum('nli,nkj->likj', BA, BA)

    # Hessian diagonal
    hess_diag[layer] += torch.einsum("ni,nj->ij", B * B, A * A)

    # KFAC right factor
    hess_kfac[layer].BB += torch.einsum("ni,nj->ij", B, B)


with autograd_lib.module_hook(compute_hess):
    autograd_lib.backward_hessian(output, loss='LeastSquares')

for layer in model.modules():
    print(hess_diag[layer])
