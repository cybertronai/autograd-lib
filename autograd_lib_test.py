from collections import defaultdict

import torch
import sys
from attrdict import AttrDefault

from autograd_lib import autograd_lib
from autograd_lib import util as u


def create_toy_model():
    """
    Create model from https://www.wolframcloud.com/obj/yaroslavvb/newton/linear-jacobians-and-hessians.nb
    PyTorch works on transposed representation, hence to obtain Y from notebook, do model(A.T).T
    """

    model: u.SimpleMLP = u.SimpleMLP([2, 2, 2], bias=False)
    autograd_lib.register(model)

    A = torch.tensor([[-1., 4], [3, 0]])
    B = torch.tensor([[-4., 3], [2, 6]])
    X = torch.tensor([[-5., 0], [-2, -6]], requires_grad=True)

    model.layers[0].weight.data.copy_(X)
    model.layers[1].weight.data.copy_(B.t())
    return A, model


def test_hessian_full():
    data_width = 3
    batch_size = 2
    d = [data_width**2, 10]
    o = d[-1]
    n = batch_size
    train_steps = 1

    model = u.SimpleMLP(d, nonlin=False, bias=True)
    autograd_lib.register(model)
    dataset = u.TinyMNIST(dataset_size=batch_size, data_width=data_width)
    trainloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    train_iter = iter(trainloader)

    loss_fn = torch.nn.CrossEntropyLoss()

    hess = defaultdict(float)
    for train_step in range(train_steps):
        data, targets = next(train_iter)

        activations = {}
        def save_activations(layer, a, _):
            activations[layer] = a

        with autograd_lib.module_hook(save_activations):
            output = model(data)
            loss = loss_fn(output, targets)

        def compute_hess(layer, _, B):
            A = activations[layer]
            BA = torch.einsum("nl,ni->nli", B, A)
            hess[layer] += torch.einsum('nli,nkj->likj', BA, BA)

        with autograd_lib.module_hook(compute_hess):
            autograd_lib.backward_hessian(output, loss='CrossEntropy', retain_graph=True)

        # compute Hessian through autograd
        H_autograd = u.hessian(loss, model.layers[0].weight)
        u.check_close(hess[model.layers[0]] / n, H_autograd)


def test_hessian_kfac():
    model: u.SimpleMLP = u.SimpleMLP([2, 2], nonlin=True, bias=True)
    model.layers[0].weight.data.copy_(torch.eye(2))
    autograd_lib.register(model)
    loss_fn = torch.nn.CrossEntropyLoss()

    data = u.to_logits(torch.tensor([[0.7, 0.3]]))
    targets = torch.tensor([0])

    data = data.repeat([3, 1])
    targets = targets.repeat([3])
    n = len(data)

    activations = {}
    hessians = defaultdict(lambda: AttrDefault(float))

    for i in range(n):
        def save_activations(layer, A, _):
            activations[layer] = A
            hessians[layer].AA += torch.einsum("ni,nj->ij", A, A)

        with autograd_lib.module_hook(save_activations):
            data_batch = data[i: i+1]
            targets_batch = targets[i: i+1]
            Y = model(data_batch)
            loss = loss_fn(Y, targets_batch)

        def compute_hess(layer, _, B):
            hessians[layer].BB += torch.einsum("ni,nj->ij", B, B)

        with autograd_lib.module_hook(compute_hess):
            autograd_lib.backward_hessian(Y, loss='CrossEntropy', retain_graph=True)

    # check diagonal entries against autograd
    hess_autograd = u.hessian(loss, model.layers[0].weight)
    hess0_factored = hessians[model.layers[0]]

    diag_autograd = torch.einsum('lili->li', hess_autograd)
    diag_kfac = torch.einsum('ll,ii->li', hess0_factored.BB / n, hess0_factored.AA / n)
    u.check_close(diag_autograd,  diag_kfac)

    # check all entries against autograd
    hess0 = torch.einsum('kl,ij->kilj', hess0_factored.BB / n, hess0_factored.AA / n)
    u.check_close(hess_autograd, hess0)


def test_hessian_diag():
    """Test regular and diagonal Hessian computation against autograd"""
    data_width = 3
    batch_size = 2
    d = [data_width**2, 6, 10]
    train_steps = 2

    model: u.SimpleMLP = u.SimpleMLP(d, nonlin=True, bias=True)
    autograd_lib.register(model)
    dataset = u.TinyMNIST(dataset_size=batch_size*train_steps, data_width=data_width)
    trainloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    train_iter = iter(trainloader)

    loss_fn = torch.nn.CrossEntropyLoss()

    for train_step in range(train_steps):
        data, targets = next(train_iter)

        hess = defaultdict(float)
        hess_diag = defaultdict(float)
        activations = {}
        def save_activations(layer, a, _):
            activations[layer] = a

        with autograd_lib.module_hook(save_activations):
            output = model(data)
            loss = loss_fn(output, targets)

        def compute_hess(layer, _, B):
            A = activations[layer]
            BA = torch.einsum("nl,ni->nli", B, A)
            hess[layer] += torch.einsum('nli,nkj->likj', BA, BA)
            hess_diag[layer] += torch.einsum("ni,nj->ij", B * B, A * A)

        with autograd_lib.module_hook(compute_hess):
            autograd_lib.backward_hessian(output, loss='CrossEntropy', retain_graph=True)

        # compute Hessian through autograd
        # for layer in model.layers:

        for layer in model.layers:
            H_autograd = u.hessian(loss, layer.weight)
            u.check_close(hess[layer] / batch_size, H_autograd)
            diag_autograd = torch.einsum('lili->li', H_autograd)
            u.check_close(diag_autograd, hess_diag[layer]/batch_size)


def test_hessian_diag_sqr():
    """Like above, but using LeastSquares loss"""
    
    data_width = 3
    batch_size = 2
    d = [data_width**2, 6, 10]
    train_steps = 2

    model: u.SimpleMLP = u.SimpleMLP(d, nonlin=True, bias=True)
    autograd_lib.register(model)
    dataset = u.TinyMNIST(dataset_size=batch_size*train_steps, data_width=data_width)
    trainloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    train_iter = iter(trainloader)

    #loss_fn = torch.nn.CrossEntropyLoss()
    loss_fn = u.least_squares

    for train_step in range(train_steps):
        data, targets = next(train_iter)

        hess = defaultdict(float)
        hess_diag = defaultdict(float)
        activations = {}
        def save_activations(layer, a, _):
            activations[layer] = a

        with autograd_lib.module_hook(save_activations):
            output = model(data)
            loss = loss_fn(output, torch.zeros_like(output))

        def compute_hess(layer, _, B):
            A = activations[layer]
            BA = torch.einsum("nl,ni->nli", B, A)
            hess[layer] += torch.einsum('nli,nkj->likj', BA, BA)
            hess_diag[layer] += torch.einsum("ni,nj->ij", B * B, A * A)

        with autograd_lib.module_hook(compute_hess):
            autograd_lib.backward_hessian(output, loss='LeastSquares', retain_graph=True)

        # compute Hessian through autograd
        # for layer in model.layers:

        for layer in model.layers:
            H_autograd = u.hessian(loss, layer.weight)
            u.check_close(hess[layer] / batch_size, H_autograd)
            diag_autograd = torch.einsum('lili->li', H_autograd)
            u.check_close(diag_autograd, hess_diag[layer]/batch_size)


def test_jacobian_full():
    """Test computing Jacobian squared"""
    A, model = create_toy_model()
    data = A.t()
    data = data.repeat(3, 1)

    activations = {}
    jacobians = defaultdict(float)
    def save_activations(layer, a, _):
        activations[layer] = a
    def compute_hessian(layer, _, B):
        A = activations[layer]
        BA = torch.einsum("nl,ni->nli", B, A)
        jacobians[layer] += torch.einsum('nli,nkj->likj', BA, BA)

    for x in data:
        with autograd_lib.module_hook(save_activations):
            y = model(x)
            loss = torch.sum(y * y) / 2

        with autograd_lib.module_hook(compute_hessian):
            autograd_lib.backward_jacobian(y, retain_graph=True)

    J0 = jacobians[model.layers[0]]

    # check result against autograd
    J = u.jacobian(model(data), model.layers[0].weight)
    J_autograd = torch.einsum('noij,nokl->ijkl', J, J)
    u.check_equal(J0, J_autograd)

    # jacobian squared is equal to Hessian
    loss = u.least_squares(model(data), aggregation='sum')
    u.check_equal(J0, u.hessian(loss, model.layers[0].weight))


def test_fisher_full():
    torch.set_default_dtype(torch.float64)
    u.seed_random(1)
    A, model = create_toy_model()

    activations = {}

    def save_activations(layer, a, _):
        if layer != model.layers[0]:
            return
        activations[layer] = a

    fisher = [0]

    def compute_fisher(layer, _, B):
        if layer != model.layers[0]:
            return
        A = activations[layer]
        n = A.shape[0]

        Jo = torch.einsum("ni,nj->nij", B, A).reshape(n, -1)
        fisher[0] += torch.einsum('ni,nj->ij', Jo, Jo)

    for x in A.t():
        with autograd_lib.module_hook(save_activations):
            y = model(x)
            loss = torch.sum(y * y) / 2

        with autograd_lib.module_hook(compute_fisher):
            loss.backward()

    # result computed using single step forward prop
    result0 = torch.tensor([[5.383625e+06, -3.675000e+03, 4.846250e+06, -6.195000e+04],
                            [-3.675000e+03, 1.102500e+04, -6.195000e+04, 1.858500e+05],
                            [4.846250e+06, -6.195000e+04, 4.674500e+06, -1.044300e+06],
                            [-6.195000e+04, 1.858500e+05, -1.044300e+06, 3.132900e+06]])
    u.check_close(fisher[0], result0)


if __name__ == '__main__':
    test_hessian_diag_sqr()
    # sys.exit()
    test_jacobian_full()
    test_fisher_full()
    test_hessian_full()
    test_hessian_kfac()
    test_hessian_diag()
