from contextlib import contextmanager
from typing import List, Optional, Callable, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import util as u


class Settings(object):
    forward_hooks: List[Callable]   # forward subhooks called by the global hook
    backward_hooks: List[Callable]  # backward subhooks
    model: Optional[nn.Module]
    hook_handles: List[torch.utils.hooks.RemovableHandle]    # removal handles of global hooks registered with PyTorch

    def __init__(self):
        assert global_settings_initialized is False, "Reinitializing Settings, seems like a bug."
        self.hook_handles = []
        self.forward_hooks = []
        self.backward_hooks = []


global_settings_initialized = False
global_settings = Settings()
supported_layers = ['Linear', 'Conv2d']


def _forward_hook(layer: nn.Module, input_: Tuple[torch.Tensor], output: torch.Tensor):
    for hook in global_settings.forward_hooks:
        hook(layer, input_, output)


def _backward_hook(layer: nn.Module, _input: torch.Tensor, output: Tuple[torch.Tensor]):
    for hook in global_settings.backward_hooks:
        hook(layer, _input, output)


def _layer_type(layer: nn.Module) -> str:
    return layer.__class__.__name__


def register(model: nn.Module):
    """
    Registers given model with autograd_lib. This allows user to module_hook decorator
    """
    global global_settings

    layer: nn.Module
    for layer in model.modules():
        if _layer_type(layer) in supported_layers:
            global_settings.hook_handles.append(layer.register_forward_hook(_forward_hook))
            layer.register_backward_hook(_backward_hook)   # don't save handle, https://github.com/pytorch/pytorch/issues/25723


@contextmanager
def module_hook(hook: Callable):
    """Context manager for running given hook on forward or backward."""

    # TODO(y): use weak ref for the hook handles so they are removed when model goes out of scope
    assert global_settings.hook_handles, "Global hooks have not been registered. Make sure to call .register(model) on your model"
    forward_hook_called = [False]
    backward_hook_called = [False]

    def forward_hook(layer: nn.Module, input_: Tuple[torch.Tensor], output: torch.Tensor):
        assert len(input_) == 1, "Only support single input modules on forward."
        assert type(output) == torch.Tensor, "Only support single output modules on forward."
        activations = input_[0].detach()
        hook(layer, activations, output)
        forward_hook_called[0] = True

    def backward_hook(layer: nn.Module, input_: Tuple[torch.Tensor], output: Tuple[torch.Tensor]):
        assert len(output) == 1, "Only support single output modules on backward."
        backprops = output[0].detach()
        hook(layer, input_, backprops)
        backward_hook_called[0] = True

    global_settings.forward_hooks.append(forward_hook)
    global_settings.backward_hooks.append(backward_hook)
    yield
    assert forward_hook_called[0] or backward_hook_called[0], "Hook was called neither on forward nor backward pass, did you register your model?"
    assert not (forward_hook_called[0] and backward_hook_called[0]), "Hook was called both on forward and backward pass, did you register your model?"
    global_settings.forward_hooks.pop()
    global_settings.backward_hooks.pop()


def backward_jacobian(output: torch.Tensor, retain_graph=False) -> None:
    """
    Utility to compute Jacobian with respect to given output tensor. Backpropagates a row of identity matrix
    for each output of tensor. Rows are replicated across batch dimension.

    Args:
        output: target of backward
        retain_graph: same meaning as PyTorch retain_graph
    """

    assert u.is_matrix(output), "Only support rank-2 outputs."""
    # assert strategy in ('exact', 'sampled')

    n, o = output.shape
    id_mat = torch.eye(o).to(output.device)
    for idx in range(o):
        output.backward(torch.stack([id_mat[idx]] * n), retain_graph=(retain_graph or idx < o - 1))


def backward_hessian(output, loss='CrossEntropy', retain_graph=False) -> None:
    assert loss in ('CrossEntropy',), f"Only CrossEntropy loss is supported, got {loss}"
    assert u.is_matrix(output)

    # use Cholesky-like decomposition from https://www.wolframcloud.com/obj/yaroslavvb/newton/square-root-formulas.nb
    n, o = output.shape
    p = F.softmax(output, dim=1)

    mask = torch.eye(o).to(output.device).expand(n, o, o)
    diag_part = p.sqrt().unsqueeze(2).expand(n, o, o) * mask
    hess_sqrt = diag_part - torch.einsum('ij,ik->ijk', p.sqrt(), p)   # n, o, o

    for out_idx in range(o):
        output.backward(hess_sqrt[:, out_idx, :], retain_graph=(retain_graph or out_idx < o - 1))
