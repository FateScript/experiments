#!/usr/bin/env python3

import os
import pickle

import numpy as np

from mpi import (  # isort:skip
    all_to_all_array,
    auto_split,
    all_reduce,
    get_world_size,
    init_env,
    mpi_frame,
)


def softmax(x, axis=-1):
    exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def linear_transform(x, weight):
    return np.dot(x, weight.transpose())


def split_heads(x, num_heads: int):
    batch, seq, dim = x.shape
    tensor_dim = dim // num_heads
    x = x.reshape(batch, seq, num_heads, tensor_dim)
    x = np.swapaxes(x, 1, 2)  # (batch, num_heads, seq_len, tensor_dim)
    return x


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

        h = self.num_heads
        q, k, v = split_heads(q, h), split_heads(k, h), split_heads(v, h)
        out = self.attn(q, k, v, mask=mask)
        out = np.swapaxes(out, 1, 2).reshape(batch, seq, dim)
        return linear_transform(out, self.out_proj)


def mha_load_state_dict(module, state_dict):
    module.q_proj = state_dict["q_proj.weight"].numpy()
    module.k_proj = state_dict["k_proj.weight"].numpy()
    module.v_proj = state_dict["v_proj.weight"].numpy()
    module.out_proj = state_dict["out_proj.weight"].numpy()
    return module


def check_np_mha():
    # load from torch model, see experiments/nn/module/mha.py
    load_file = "mha_input_output.pkl"
    if not os.path.exists(load_file):
        print(f"File {load_file} not found. Please use nn/module/mha.py to generate it.")

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

    The num of heads should be divivied by num parallel.
    For example, if num_heads=4, num_parallel=2, then each device will handle 2 heads.
    """

    def __init__(self, q_proj, k_proj, v_proj, out_proj, num_heads: int):
        self.q_proj = q_proj
        self.k_proj = k_proj
        self.v_proj = v_proj
        self.out_proj = out_proj
        self.num_heads = num_heads
        self.attn = ScaledDotProductAttn()

    def __call__(self, x, mask=None):
        batch, _, dim = x.shape
        x = auto_split(x, axis=1)  # split by sequence length
        q = linear_transform(x, self.q_proj)
        k = linear_transform(x, self.k_proj)
        v = linear_transform(x, self.v_proj)

        h = self.num_heads
        q, k, v = split_heads(q, h), split_heads(k, h), split_heads(v, h)

        # NOTE: q, k, v are splited in head dimension.
        # The first version I implemented in commit `6ef066864d6d` split tensor in last dimension,
        # and that version requires one more all_reduce_sum operation on attn matrix.
        q = all_to_all_array(q, split_axis=1, concat_axis=-2)
        k = all_to_all_array(k, split_axis=1, concat_axis=-2)
        v = all_to_all_array(v, split_axis=1, concat_axis=-2)

        out = self.attn(q, k, v, mask=mask)
        out = all_to_all_array(out, split_axis=-2, concat_axis=1)

        out = np.swapaxes(out, 1, 2).reshape(batch, -1, dim)
        return linear_transform(out, self.out_proj)


class TensorParallelMHA:
    """
    A megatron tensor parallel version of MultiHeadAttn.
    Paper: https://arxiv.org/pdf/2205.05198

    The num of heads should be divivied by num parallel.
    For example, if num_heads=4, num_parallel=2, then each devices will handle 2 heads.
    """

    def __init__(self, q_proj, k_proj, v_proj, out_proj, num_heads: int, reduce_output=True):
        # NOTE: for two paralleled Linear, it should always be column parallel + row parallel.
        self.q_proj = auto_split(q_proj, axis=0)  # actually, it's a column parallel
        self.k_proj = auto_split(k_proj, axis=0)
        self.v_proj = auto_split(v_proj, axis=0)
        self.out_proj = auto_split(out_proj, axis=-1)  # actually, it's a row parallel
        self.num_heads = num_heads
        world_size = get_world_size()
        assert num_heads % world_size == 0, "num_heads should be divisible by num_parallel"
        self.local_heads = num_heads // get_world_size()
        self.attn = ScaledDotProductAttn()
        self.reduce_output = reduce_output

    def __call__(self, x, mask=None):
        batch, seq, _ = x.shape
        q = linear_transform(x, self.q_proj)
        k = linear_transform(x, self.k_proj)
        v = linear_transform(x, self.v_proj)

        h = self.local_heads
        q, k, v = split_heads(q, h), split_heads(k, h), split_heads(v, h)

        out = self.attn(q, k, v, mask=mask)

        out = np.swapaxes(out, 1, 2).reshape(batch, seq, -1)
        out = linear_transform(out, self.out_proj)
        if self.reduce_output:
            out = all_reduce(out, op="sum")
        return out


def load_from_file():
    load_file = "mha_input_output.pkl"
    if not os.path.exists(load_file):
        print(f"File {load_file} not found. Please use nn/module/mha.py to generate it.")

    with open(load_file, "rb") as f:
        data_dict = pickle.load(f)

    state_dict = data_dict["model_state"]
    q_proj = state_dict["q_proj.weight"].numpy()
    k_proj = state_dict["k_proj.weight"].numpy()
    v_proj = state_dict["v_proj.weight"].numpy()
    out_proj = state_dict["out_proj.weight"].numpy()
    input_data = data_dict["input"].detach().numpy()
    output_data = data_dict["output"].detach().numpy()
    return q_proj, k_proj, v_proj, out_proj, input_data, output_data


def seq_mha_forward(rank, world_size, queue, signal_queue, pipe_pairs):
    init_env(rank, world_size, queue, signal_queue, pipe_pairs)

    q_proj, k_proj, v_proj, out_proj, input_data, output_data = load_from_file()
    mha = SequenceParallelMHA(
        q_proj=q_proj,
        k_proj=k_proj,
        v_proj=v_proj,
        out_proj=out_proj,
        num_heads=4,
    )

    output_data = auto_split(output_data, axis=1)

    output = mha(input_data)
    match = np.allclose(output_data, output, atol=1e-7)
    result = "passed" if match else "failed"
    print(f"Check Seq Parallel MHA forward at rank {rank}: {result}")


def check_seq_parallel_mha():
    for world_size in [2, 4]:  # num parallel should be divisible by num_heads
        print(f"\nCheck Seq Parallel MHA with world size {world_size}")
        mpi_frame(seq_mha_forward, world_size=world_size)
        print()


def tensor_mha_forward(rank, world_size, queue, signal_queue, pipe_pairs):
    init_env(rank, world_size, queue, signal_queue, pipe_pairs)

    q_proj, k_proj, v_proj, out_proj, input_data, output_data = load_from_file()
    mha = TensorParallelMHA(
        q_proj=q_proj,
        k_proj=k_proj,
        v_proj=v_proj,
        out_proj=out_proj,
        num_heads=4,
        reduce_output=False,
    )

    output = mha(input_data)
    output = all_reduce(output, op="sum")
    match = np.allclose(output_data, output, atol=1e-7)
    result = "passed" if match else "failed"
    print(f"Check Tensor Parallel MHA forward at rank {rank}: {result}")


def check_tensor_parallel_mha():
    for world_size in [2, 4]:  # num parallel should be divisible by num_heads
        print(f"\nCheck Tensor Parallel MHA with world size {world_size}")
        mpi_frame(tensor_mha_forward, world_size=world_size)
        print()


if __name__ == "__main__":
    check_np_mha()
    check_seq_parallel_mha()
    check_tensor_parallel_mha()
