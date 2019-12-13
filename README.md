# autograd_lib

By Yaroslav Bulatov, Kazuki Osawa

Library to simplify gradient computations in PyTorch.

# example 1: per-example gradient norms

Example of using it to compute per-example gradient norms for linear layers, using trick from https://arxiv.org/abs/1510.01799

See `example_norms.py` for a runnable example. The important parts:


```
!pip install autograd-lib

from autograd_lib import autograd_lib

model = ...
loss_fn = ...
data = ...
autograd_lib.register(model)


activations = {}
backprops = {}
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

```

# Example 2: Hessian quantities

To compute exact Hessian, Hessian diagonal and KFAC approximation for all linear layers of a ReLU network in a single pass.

See `example_hessian.py` for a self-contained example. The important parts:


```
!pip install autograd-lib

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
    autograd_lib.backward_hessian(output, loss='CrossEntropy')
```

Variations:

- `autograd_lib.backward_hessian` for Hessian
- `autograd_lib.backward_jacobian` for Jacobian squared
- `loss.backward()` for empirical Fisher Information Matrix


See autograd_lib_test.py for correctness checks against PyTorch autograd.
