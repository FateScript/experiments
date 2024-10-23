#!/usr/bin/env python3

import pickle

import numpy as np

from mpi import (  # isort:skip
    all_to_all_array,
    all_reduce,
    auto_split,
    init_env,
    mpi_frame,
)


def softmax(x, axis=-1):
    exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def linear_transform(x, weight):
    return np.dot(x, weight.transpose())


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
        q = linear_transform(x, self.q_proj)
        k = linear_transform(x, self.k_proj)
        v = linear_transform(x, self.v_proj)

        q, k, v = self.split_heads(q), self.split_heads(k), self.split_heads(v)
        out = self.attn(q, k, v, mask=mask)
        out = np.swapaxes(out, 1, 2).reshape(batch, seq, dim)
        return linear_transform(out, self.out_proj)

    def split_heads(self, x):
        batch, seq, dim = x.shape
        tensor_dim = dim // self.num_heads
        x = x.reshape(batch, seq, self.num_heads, tensor_dim)
        x = np.swapaxes(x, 1, 2)  # (batch, num_heads, seq_len, tensor_dim)
        return x


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
    mha = MultiHeadAttn(32, num_heads=4)
    mha = mha_load_state_dict(mha, data_dict["model_state"])

    input_data = data_dict["input"].detach().numpy()
    output_data = data_dict["output"].detach().numpy()
    output = mha(input_data)
    match = np.allclose(output_data, output, atol=1e-7)
    result = "passed" if match else "failed"
    print(f"Check MHA forward: {result}")


class SequenceParallelMHA:
    """
    A sequence parallel version of MultiHeadAttn.
    Reference: deepspeed ulysses.
    Paper: https://arxiv.org/pdf/2309.14509
    """

    def __init__(self, q_proj, k_proj, v_proj, out_proj, num_heads: int):
        self.q_proj = q_proj
        self.k_proj = k_proj
        self.v_proj = v_proj
        self.out_proj = out_proj
        self.num_heads = num_heads

    def __call__(self, x, mask=None):
        batch, _, dim = x.shape
        x = auto_split(x, axis=1)  # split by sequence length
        q = linear_transform(x, self.q_proj)
        k = linear_transform(x, self.k_proj)
        v = linear_transform(x, self.v_proj)

        q, k, v = self.split_heads(q), self.split_heads(k), self.split_heads(v)

        q = all_to_all_array(q, split_axis=-1, concat_axis=-2)
        k = all_to_all_array(k, split_axis=-1, concat_axis=-2)
        v = all_to_all_array(v, split_axis=-1, concat_axis=-2)

        out = self.parallel_attn(q, k, v, mask=mask)
        out = all_to_all_array(out, split_axis=-2, concat_axis=-1)

        out = np.swapaxes(out, 1, 2).reshape(batch, -1, dim)
        return linear_transform(out, self.out_proj)

    def parallel_attn(self, q, k, v, mask=None):
        _, head, _, dim = q.shape  # usually (batch, seq_len, dim) or (batch, head, length, dim)
        scale = np.sqrt(dim * head)
        mat = np.matmul(q, np.swapaxes(k, -1, -2)) / scale

        # Each device only has a part of the mat like
        # [w_0 * x0], [w1 * x1], [w_2 * x2], ...
        # the true attn matrix is the sum of them, so all_reduce is needed.
        # This is the only difference between parallel mha and normal mha.
        mat = all_reduce(mat, op="sum")

        if mask is not None:
            mat = np.where(mask == 0, -1e6, mat)

        mat = softmax(mat, axis=-1)

        v = np.matmul(mat, v)
        return v

    def split_heads(self, x):
        batch, seq, dim = x.shape
        tensor_dim = dim // self.num_heads
        x = x.reshape(batch, seq, self.num_heads, tensor_dim)
        x = np.swapaxes(x, 1, 2)  # (batch, num_heads, seq_len, tensor_dim)
        return x


def seq_mha_forward(rank, world_size, queue, signal_queue, pipe_pairs):
    init_env(rank, world_size, queue, signal_queue, pipe_pairs)

    load_file = "mha_input_output.pkl"
    with open(load_file, "rb") as f:
        data_dict = pickle.load(f)

    state_dict = data_dict["model_state"]
    q_proj = state_dict["q_proj.weight"].numpy()
    k_proj = state_dict["k_proj.weight"].numpy()
    v_proj = state_dict["v_proj.weight"].numpy()
    out_proj = state_dict["out_proj.weight"].numpy()

    mha = SequenceParallelMHA(
        q_proj=q_proj,
        k_proj=k_proj,
        v_proj=v_proj,
        out_proj=out_proj,
        num_heads=4,
    )

    input_data = data_dict["input"].detach().numpy()
    output_data = data_dict["output"].detach().numpy()
    output_data = auto_split(output_data, axis=1)

    output = mha(input_data)
    match = np.allclose(output_data, output, atol=1e-7)
    result = "passed" if match else "failed"
    print(f"Check Seq Parallel MHA forward at rank {rank}: {result}")


def check_seq_parallel_mha():
    world_size = 4
    mpi_frame(seq_mha_forward, world_size=world_size)


if __name__ == "__main__":
    check_np_mha()
    check_seq_parallel_mha()
