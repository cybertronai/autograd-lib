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

depth = 5
width = 2
n = 3
model = simple_model(width, depth)
data = torch.ones((n, width))
targets = torch.ones((n, width))
loss_fn = least_squares

from autograd_lib import autograd_lib
autograd_lib.register(model)

activations = {}
norms = [torch.zeros(n)]

def save_activations(layer, A, _):
    activations[layer] = A
    
with autograd_lib.module_hook(save_activations):
    output = model(data)
    loss = loss_fn(output)

def per_example_norms(layer, _, B):
    A = activations[layer]
    norms[0]+=(A*A).sum(dim=1)*(B*B).sum(dim=1)

with autograd_lib.module_hook(per_example_norms):
    loss.backward()

print('per-example gradient norms squared:', norms[0])
