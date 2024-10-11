#!/usr/bin/env python3

import functools
import os

import numpy as np

from mpi import (  # isort:skip
    all_reduce,
    all_gather,
    barrier,
    init_env,
    mpi_frame,
    split_first_dim,
    split_last_dim,
)

__all__ = [
    "Linear",
    "ColumnParallelLinear",
    "RowParallelLinear",
]


class Linear:

    def __init__(self, weights, bias=None):
        self.weight = weights
        self.bias = bias
        self.context = {}
        self.grad = {}

    def forward(self, x):
        self.context["x"] = x
        y = np.dot(x, self.weight)
        if self.bias:
            y += self.bias
        return y

    def backward(self, grad):
        # y = x * w + b

        # partial y / partial x = w.T
        # partial y / partial w = x.T
        # partial y / partial b = 1

        # grad_w = x.T * grad
        # grad_x = grad * w.T
        # grad_b = sum(grad)  # first dim
        input_data = self.context.pop("x")

        if self.bias:
            grad_bias = np.sum(grad, axis=0)
            self.grad["bias"] = grad_bias

        grad_weight = np.dot(input_data.T, grad)
        self.grad["weight"] = grad_weight

        prev_layer_grad = np.dot(grad, self.weight.T)
        return prev_layer_grad

    def step_grad(self, learning_rate: float):
        grad_weight = self.grad.pop("weight")
        self.weight -= learning_rate * grad_weight
        if self.bias:
            grad_bias = self.grad.pop("bias")
            self.bias -= learning_rate * grad_bias


class ColumnParallelLinear(Linear):
    """
    Linear layer with column parallelism.

    The linear layer is defined as y = x * w + b. w is parallelized along
    its second dimension as w = [w_1, ..., w_p].
    """
    def __init__(self, weights, bias=None, gather_output: bool = True):
        super().__init__(weights, bias)
        self.gather_output = gather_output

    def master_weight(self):
        barrier()
        weights = all_gather(self.weight)
        return weights

    def forward(self, x):
        y = super().forward(x)
        if self.gather_output:
            y = all_gather(y)
        return y

    def backward(self, grad):
        if self.gather_output:
            grad = split_last_dim(grad)
        data_grad = super().backward(grad)
        data_grad = all_reduce(data_grad, op="sum")
        return data_grad


class RowParallelLinear(Linear):
    """
    Linear layer with row parallelism.

    The linear layer is defined as y = x * w + b. w is parallelized along
    its first dimension and x along its second dimension as:
               -   -
              | w_1 |
              | .   |
          w = | .   |        x = [x_1, ..., x_p]
              | .   |
              | w_p |
               -   -
    """
    def __init__(self, weights, bias=None, input_is_parallel: bool = False):
        """
        Args:
            input_is_parallel(bool): whether the input x is already parallelized along the last dim
        """
        super().__init__(weights, bias)
        self.input_is_parallel = input_is_parallel

    def forward(self, x):
        if self.input_is_parallel:
            data = x
        else:
            data = split_last_dim(x)

        part_y = super().forward(data)
        y = all_reduce(part_y, op="sum")
        return y

    def backward(self, grad):
        data_grad = super().backward(grad)
        if self.input_is_parallel:
            return data_grad
        data_grad = all_gather(data_grad)
        return data_grad


def linear_step():
    np.random.seed(42)

    batch_size = 16
    input_dim, output_dim = 128, 32
    lr = 1

    w = np.random.rand(input_dim, output_dim)
    data = np.random.rand(batch_size, input_dim)
    linear = Linear(weights=w)
    output = linear.forward(data)
    grad_output = np.random.rand(*output.shape)
    data_grad = linear.backward(grad_output)
    linear.step_grad(lr)

    np.save("linear_weight.npy", linear.weight)
    np.save("linear_output.npy", output)
    np.save("linear_data_grad.npy", data_grad)


def col_linear(rank, world_size, queue, signal_queue, pipe_pairs):
    init_env(rank, world_size, queue, signal_queue, pipe_pairs)
    np.random.seed(42)

    batch_size = 16
    input_dim, output_dim = 128, 32
    lr = 1

    w = np.random.rand(input_dim, output_dim)
    data = np.random.rand(batch_size, input_dim)

    w_parallel = split_last_dim(w)
    print(f"rank {rank} w_parallel shape: {w_parallel.shape}")
    linear = ColumnParallelLinear(weights=w_parallel, gather_output=True)

    output = linear.forward(data)
    grad_output = np.random.rand(*output.shape)
    data_grad = linear.backward(grad_output)
    linear.step_grad(lr)

    np.save(f"linear_weight_rank{rank}.npy", linear.weight)
    np.save(f"linear_output_rank{rank}.npy", output)
    np.save(f"linear_data_grad_rank{rank}.npy", data_grad)


def row_linear(rank, world_size, queue, signal_queue, pipe_pairs):
    init_env(rank, world_size, queue, signal_queue, pipe_pairs)
    np.random.seed(42)

    batch_size = 16
    input_dim, output_dim = 128, 32
    lr = 1

    w = np.random.rand(input_dim, output_dim)
    data = np.random.rand(batch_size, input_dim)

    w_parallel = split_first_dim(w)
    print(f"rank {rank} w_parallel shape: {w_parallel.shape}")
    linear = RowParallelLinear(weights=w_parallel)

    output = linear.forward(data)
    grad_output = np.random.rand(*output.shape)
    data_grad = linear.backward(grad_output)
    linear.step_grad(lr)

    np.save(f"linear_weight_rank{rank}.npy", linear.weight)
    np.save(f"linear_output_rank{rank}.npy", output)
    np.save(f"linear_data_grad_rank{rank}.npy", data_grad)


def col_parallel_step():
    world_size = 4
    mpi_frame(col_linear, world_size=world_size)


def row_parallel_step():
    world_size = 4
    mpi_frame(row_linear, world_size=world_size)


def check(prefix: str):
    print(f"\nchecking {prefix}...")
    out = np.load(f"{prefix}.npy")
    for r in range(8):
        load_name = f"{prefix}_rank{r}.npy"
        if os.path.exists(load_name):
            data = np.load(load_name)
            assert np.allclose(out, data)

    print(f"checking {prefix} passed.")


def check_weights(para_type: str = "col"):
    assert para_type in ["col", "row"]
    axis = -1 if para_type == "col" else 0
    print(f"\nchecking {para_type} weights...")
    out = np.load("linear_weight.npy")

    weights = []
    for r in range(8):
        load_name = f"linear_weight_rank{r}.npy"
        if os.path.exists(load_name):
            data = np.load(load_name)
            weights.append(data)
    w = np.concatenate(weights, axis=axis)
    assert np.allclose(out, w, atol=1e-8)

    print(f"checking {para_type} weights passed.")


check_output = functools.partial(check, prefix="linear_output")
check_grad = functools.partial(check, prefix="linear_data_grad")


def test_col_linear():
    linear_step()
    col_parallel_step()

    check_output()
    check_grad()
    check_weights(para_type="col")


def test_row_linear():
    linear_step()
    row_parallel_step()

    check_output()
    check_grad()
    check_weights(para_type="row")


if __name__ == "__main__":
    print("Test linear layer with column parallelism...")
    test_col_linear()

    print("\n\nTest linear layer with row parallelism...")
    test_row_linear()
