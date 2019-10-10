# Take simple example, plot per-layer stats over time
import random
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torchvision.datasets as datasets
from PIL import Image

_pytorch_floating_point_types = (torch.float16, torch.float32, torch.float64)

_numpy_type_map = {
    'float64': torch.DoubleTensor,
    'float32': torch.FloatTensor,
    'float16': torch.HalfTensor,
    'int64': torch.LongTensor,
    'int32': torch.IntTensor,
    'int16': torch.ShortTensor,
    'int8': torch.CharTensor,
    'uint8': torch.ByteTensor,
}


def pytorch_dtype_to_floating_numpy_dtype(dtype):
    """Converts PyTorch dtype to numpy floating point dtype, defaulting to np.float32 for non-floating point types."""
    if dtype == torch.float64:
        dtype = np.float64
    elif dtype == torch.float32:
        dtype = np.float32
    elif dtype == torch.float16:
        dtype = np.float16
    else:
        dtype = np.float32
    return dtype


def to_numpy(x, dtype: np.dtype = None) -> np.ndarray:
    """
    Convert numeric object to floating point numpy array. If dtype is not specified, use PyTorch default dtype.

    Args:
        x: numeric object
        dtype: numpy dtype, must be floating point

    Returns:
        floating point numpy array
    """

    assert np.issubdtype(dtype, np.floating), "dtype must be real-valued floating point"

    # Convert to normal_form expression from a special form (https://reference.wolfram.com/language/ref/Normal.html)
    if hasattr(x, 'normal_form'):
        x = x.normal_form()

    if type(x) == np.ndarray:
        assert np.issubdtype(x.dtype, np.floating), f"numpy type promotion not implemented for {x.dtype}"

    if type(x) == torch.Tensor:
        dtype = pytorch_dtype_to_floating_numpy_dtype(x.dtype)
        return x.detach().cpu().numpy().astype(dtype)

    # list or tuple, iterate inside to convert PyTorch arrrays
    if type(x) in [list, tuple]:
        x = [to_numpy(r) for r in x]

    # Some Python type, use numpy conversion
    result = np.array(x, dtype=dtype)
    assert np.issubdtype(result.dtype, np.number), f"Provided object ({result}) is not numeric, has type {result.dtype}"
    if dtype is None:
        return result.astype(pytorch_dtype_to_floating_numpy_dtype(torch.get_default_dtype()))
    return result


def to_numpys(*xs, dtype=np.float32):
    return (to_numpy(x, dtype) for x in xs)


def check_equal(observed, truth, rtol=1e-9, atol=1e-12, label: str= '') -> None:
    """
    Assert fail any entries in two arrays are not close to each to desired tolerance. See np.allclose for meaning of rtol, atol

    """

    truth = to_numpy(truth)
    observed = to_numpy(observed)

    assert truth.shape == observed.shape, f"Observed shape {observed.shape}, expected shape {truth.shape}"
    # run np.testing.assert_allclose for extra info on discrepancies
    if not np.allclose(observed, truth, rtol=rtol, atol=atol, equal_nan=True):
        print(f'Numerical testing failed for {label}')
        np.testing.assert_allclose(truth, observed, rtol=rtol, atol=atol, equal_nan=True)


def check_close(a0, b0, rtol=1e-5, atol=1e-8, label: str= '') -> None:
    """Convenience method for check_equal with tolerances defaulting to typical errors observed in neural network
    ops in float32 precision."""
    return check_equal(a0, b0, rtol=rtol, atol=atol, label=label)


# Fork of SimpleModel that doesn't automatically register hooks, for autograd_lib.py refactoring
class SimpleModel2(nn.Module):
    """Simple sequential model. Adds layers[] attribute, flags to turn on/off hooks, and lookup mechanism from layer to parent
    model."""

    layers: List[nn.Module]
    all_layers: List[nn.Module]

    def __init__(self, *args, **kwargs):
        super().__init__()


def least_squares(data, targets=None, aggregation='mean'):
    """Least squares loss (like MSELoss, but an extra 1/2 factor."""
    assert is_matrix(data), f"Expected matrix, got {data.shape}"
    assert aggregation in ('mean', 'sum')
    if targets is None:
        targets = torch.zeros_like(data)
    err = data - targets.view(-1, data.shape[1])
    normalizer = len(data) if aggregation == 'mean' else 1
    return torch.sum(err * err) / 2 / normalizer


# Autograd functions, from https://gist.github.com/apaszke/226abdf867c4e9d6698bd198f3b45fb7
# noinspection PyTypeChecker
def jacobian(y: torch.Tensor, x: torch.Tensor, create_graph=False):
    jac = []
    flat_y = y.reshape(-1)
    grad_y = torch.zeros_like(flat_y)
    for i in range(len(flat_y)):
        grad_y[i] = 1.
        grad_x, = torch.autograd.grad(flat_y, x, grad_y, retain_graph=True, create_graph=create_graph)
        jac.append(grad_x.reshape(x.shape))
        grad_y[i] = 0.
    return torch.stack(jac).reshape(y.shape + x.shape)


def hessian(y: torch.Tensor, x: torch.Tensor):
    return jacobian(jacobian(y, x, create_graph=True), x)


def seed_random(seed: int) -> None:
    """Manually set seed to seed for configurable random number generators in current process."""
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class SimpleMLP(nn.Module):
    """Simple feedforward network that works on images."""

    layers: List[nn.Module]
    all_layers: List[nn.Module]

    def __init__(self, d: List[int], nonlin=False, bias=False):
        """
        Feedfoward network of linear layers with optional ReLU nonlinearity. Stores layers in "layers" attr, ie
        model.layers[0] refers to first linear layer.

        Args:
            d: list of layer dimensions, ie [768, 20, 10] for MNIST 10-output with hidden layer of 20
            nonlin: whether to include ReLU nonlinearity
        """
        super().__init__()
        self.layers: List[nn.Module] = []
        self.all_layers: List[nn.Module] = []
        self.d: List[int] = d
        for i in range(len(d) - 1):
            linear = nn.Linear(d[i], d[i + 1], bias=bias)
            setattr(linear, 'name', f'{i:02d}-linear')
            self.layers.append(linear)
            self.all_layers.append(linear)
            if nonlin:
                self.all_layers.append(nn.ReLU())
        self.predict = torch.nn.Sequential(*self.all_layers)

    def forward(self, x: torch.Tensor):
        x = x.reshape((-1, self.d[0]))
        return self.predict(x)


def is_row_matrix(dd):
    return len(dd.shape) == 2 and dd.shape[0] == 1


def is_col_matrix(dd):
    return len(dd.shape) == 2 and dd.shape[1] == 1


def is_square_matrix(dd):
    return len(dd.shape) == 2 and dd.shape[0] == dd.shape[1] and dd.shape[0] >= 1


def is_vector(dd) -> bool:
    shape = dd.shape
    return len(shape) == 1 and shape[0] >= 1


def is_matrix(dd) -> bool:
    shape = dd.shape
    return len(shape) == 2 and shape[0] >= 1 and shape[1] >= 1


def to_logits(p: torch.Tensor) -> torch.Tensor:
    """Inverse of F.softmax"""
    if len(p.shape) == 1:
        batch = torch.unsqueeze(p, 0)
    else:
        assert len(p.shape) == 2
        batch = p

    batch = torch.log(batch) - torch.log(batch[:, -1])
    return batch.reshape(p.shape)


class TinyMNIST(datasets.MNIST):
    """Custom-size MNIST dataset for testing. Generates data/target images with reduced resolution and 0
    channels. When provided with original 28, 28 resolution, generates standard 1 channel MNIST dataset.
    """

    def __init__(self, dataset_root='/tmp/data', data_width=4, targets_width=4, dataset_size=0,
                 train=True):
        """

        Args:
            data_width: dimension of input images
            targets_width: dimension of target images
            dataset_size: number of examples, use for smaller subsets and running locally
        """
        super().__init__(dataset_root, download=True, train=train)

        original_targets = True

        if dataset_size > 0:
            self.data = self.data[:dataset_size, :, :]
            self.targets = self.targets[:dataset_size]

        if data_width != 28 or targets_width != 28:
            new_data = np.zeros((self.data.shape[0], data_width, data_width))
            new_targets = np.zeros((self.data.shape[0], targets_width, targets_width))
            for i in range(self.data.shape[0]):
                arr = self.data[i, :].numpy().astype(np.uint8)
                im = Image.fromarray(arr)
                im.thumbnail((data_width, data_width), Image.ANTIALIAS)
                new_data[i, :, :] = np.array(im) / 255
                im = Image.fromarray(arr)
                im.thumbnail((targets_width, targets_width), Image.ANTIALIAS)
                new_targets[i, :, :] = np.array(im) / 255
            self.data = torch.from_numpy(new_data).type(torch.get_default_dtype())
            if not original_targets:
                self.targets = torch.from_numpy(new_targets).type(torch.get_default_dtype())
        else:
            self.data = self.data.type(torch.get_default_dtype()).unsqueeze(1)
            if not original_targets:
                self.targets = self.data

        if torch.cuda.is_available():
            self.data, self.targets = self.data.to('cuda'), self.targets.to('cuda')

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        return img, target

