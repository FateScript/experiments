#!/usr/bin/env python3

import numpy as np


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
        # y = w * x + b

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


def test_grad():
    np.random.seed(42)

    batch_size = 16
    input_dim, output_dim = 100, 10
    w = np.random.rand(input_dim, output_dim)
    data = np.random.rand(batch_size, input_dim)
    linear = Linear(weights=w)
    output = linear.forward(data)
    grad_output = np.random.rand(*output.shape)
    data_grad = linear.backward(grad_output)
    return output, data_grad, linear
    breakpoint()
    pass


if __name__ == "__main__":
    test_grad()
