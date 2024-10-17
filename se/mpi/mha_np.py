#!/usr/bin/env python3

import pickle

import numpy as np


def softmax(x, axis=-1):
    exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


class ScaledDotProductAttn:

    def __init__(self, return_attn=False):
        self.return_attn = return_attn

    def __call__(self, q, k, v, mask=None):
        *_, dim = q.shape  # usually (batch, seq_len, dim) or (batch, head, length, dim)
        scale = np.sqrt(dim)
        mat = np.matmul(q, np.swapaxes(k, -1, -2)) / scale

        if mask is not None:
            mat = np.where(mask == 0, -1e6, mat)

        mat = softmax(mat, axis=-1)

        v = np.matmul(mat, v)
        if self.return_attn:
            return v, mat
        return v


class MultiHeadAttn:

    def __init__(self, dim: int, num_heads: int):
        self.dim = dim
        self.num_heads = num_heads
        self.q_proj = self.linear_init(dim, dim)
        self.k_proj = self.linear_init(dim, dim)
        self.v_proj = self.linear_init(dim, dim)
        self.out_proj = self.linear_init(dim, dim)

        self.attn = ScaledDotProductAttn()

    def linear_init(self, in_dim, out_dim):
        return np.random.randn(in_dim, out_dim) * np.sqrt(2.0 / (in_dim + out_dim))

    def __call__(self, x, mask=None):
        batch, seq, dim = x.shape
        q = self.linear_transform(x, self.q_proj)
        k = self.linear_transform(x, self.k_proj)
        v = self.linear_transform(x, self.v_proj)

        q, k, v = self.split_heads(q), self.split_heads(k), self.split_heads(v)
        out = self.attn(q, k, v, mask=mask)
        out = np.swapaxes(out, 1, 2).reshape(batch, seq, dim)
        return self.linear_transform(out, self.out_proj)

    def split_heads(self, x):
        batch, seq, dim = x.shape
        tensor_dim = dim // self.num_heads
        x = x.reshape(batch, seq, self.num_heads, tensor_dim)
        x = np.swapaxes(x, 1, 2)  # (batch, num_heads, seq_len, tensor_dim)
        return x

    def linear_transform(self, x, weight):
        return np.dot(x, weight.transpose())


def mha_load_state_dict(module, state_dict):
    module.q_proj = state_dict["q_proj.weight"].numpy()
    module.k_proj = state_dict["k_proj.weight"].numpy()
    module.v_proj = state_dict["v_proj.weight"].numpy()
    module.out_proj = state_dict["out_proj.weight"].numpy()
    return module


def check_np_mha():
    # load from torch model, see experiments/nn/module/mha.py
    load_file = "mha_input_output.pkl"
    with open(load_file, "rb") as f:
        data_dict = pickle.load(f)
    mha = MultiHeadAttn(32, 4)
    mha = mha_load_state_dict(mha, data_dict["model_state"])

    input_data = data_dict["input"].detach().numpy()
    output_data = data_dict["output"].detach().numpy()
    output = mha(input_data)
    match = np.allclose(output_data, output, atol=1e-7)
    result = "passed" if match else "failed"
    print(f"Check MHA forward: {result}")


if __name__ == "__main__":
    check_np_mha()
