#!/usr/bin/env python3

# A numpy version implementation of Flash Attention
# reference paper: https://arxiv.org/abs/2205.14135

import itertools
import os
import pickle

import numpy as np

vec_diag = np.vectorize(np.diag, signature="(n)->(n,n)")


def linear_transform(x, weight):
    return np.dot(x, weight.transpose())


def split_heads(x, num_heads: int):
    batch, seq, dim = x.shape
    tensor_dim = dim // num_heads
    x = x.reshape(batch, seq, num_heads, tensor_dim)
    x = np.swapaxes(x, 1, 2)  # (batch, num_heads, seq_len, tensor_dim)
    return x


class FlashAttnV1:

    def __init__(self, dim: int, num_heads: int):
        self.dim = dim
        self.num_heads = num_heads
        self.q_proj = self.linear_init(dim, dim)
        self.k_proj = self.linear_init(dim, dim)
        self.v_proj = self.linear_init(dim, dim)
        self.out_proj = self.linear_init(dim, dim)

    def linear_init(self, in_dim, out_dim):
        return np.random.randn(in_dim, out_dim) * np.sqrt(2.0 / (in_dim + out_dim))

    def __call__(self, x, q_block_size: int = 2, kv_block_size: int = 1):
        batch, seq, dim = x.shape
        q = linear_transform(x, self.q_proj)
        k = linear_transform(x, self.k_proj)
        v = linear_transform(x, self.v_proj)

        h = self.num_heads
        q, k, v = split_heads(q, h), split_heads(k, h), split_heads(v, h)
        out = self.flash_attn(q, k, v, q_block_size, kv_block_size)
        out = np.swapaxes(out, 1, 2).reshape(batch, seq, dim)
        return linear_transform(out, self.out_proj)

    def flash_attn(self, q, k, v, q_block_size: int = 2, kv_block_size: int = 1):
        """Algorithm 1 in the paper.

        Args:
            q: query tensor, shape (batch, num_heads, q_len, dim)
            k: key tensor, shape (batch, num_heads, kv_len, dim)
            v: value tensor, shape (batch, num_heads, kv_len, dim)
            q_block_size(int): block size for query tensor, B_c in the paper, default to 2.
            kv_block_size(int): block size for key and value tensor, B_r in the paper, default to 1.
        """
        batch_size = q.shape[0]
        num_query = q.shape[-2] // q_block_size
        num_kv = k.shape[-2] // kv_block_size
        k_t = np.swapaxes(k, -1, -2)
        split_q = np.split(q, num_query, axis=-2)
        split_k = np.split(k_t, num_kv, axis=-1)
        split_v = np.split(v, num_kv, axis=-2)
        attn_scalar = np.sqrt(q.shape[-1])

        # init O, l, and m in algorithm 1
        # O is outputs, l is exp_sum scalar, m is row_max
        init_shape = (batch_size, self.num_heads, q_block_size, 1)
        outputs = [np.zeros_like(split_q[0]) for _ in range(num_query)]
        exp_sum = [np.zeros(init_shape) for _ in range(num_query)]
        row_max = [np.ones(init_shape) * float("-inf") for _ in range(num_query)]

        for local_k, local_v in zip(split_k, split_v):  # outer loop of key, value
            for q_idx, local_q in enumerate(split_q):  # inner loop of query
                prev_row_max = row_max[q_idx]
                prev_exp_sum = exp_sum[q_idx]
                prev_outputs = outputs[q_idx]

                # NOTE: don't forget to divide by sqrt(d_k)
                s_ij = local_q @ local_k / attn_scalar
                local_max = np.max(s_ij, axis=-1, keepdims=True)
                p_ij = np.exp(s_ij - local_max)
                local_exp_sum = np.sum(p_ij, axis=-1, keepdims=True)

                update_max = np.maximum(prev_row_max, local_max)
                history_scalar = np.exp(prev_row_max - update_max)
                current_scalar = np.exp(local_max - update_max)

                # compute output
                o_ij = p_ij @ local_v
                prev_diag = vec_diag(prev_exp_sum.squeeze(-1))
                update_outputs = prev_diag @ (history_scalar * prev_outputs) + current_scalar * o_ij  # noqa
                update_exp_sum = history_scalar * prev_exp_sum + current_scalar * local_exp_sum
                diag_inverse = vec_diag(1 / update_exp_sum.squeeze(-1))  # 1 / x to get inverse
                update_outputs = diag_inverse @ update_outputs

                # update
                row_max[q_idx] = update_max
                exp_sum[q_idx] = update_exp_sum
                outputs[q_idx] = update_outputs

        return np.concatenate(outputs, axis=-2)


def flash_attn_load_state_dict(module, state_dict):
    module.q_proj = state_dict["q_proj.weight"].numpy()
    module.k_proj = state_dict["k_proj.weight"].numpy()
    module.v_proj = state_dict["v_proj.weight"].numpy()
    module.out_proj = state_dict["out_proj.weight"].numpy()
    return module


def check_np_flash_attn_v1():
    # load from torch model, see experiments/nn/module/mha.py
    load_file = "mha_input_output.pkl"
    if not os.path.exists(load_file):
        print(f"File {load_file} not found. Please use nn/module/mha.py to generate it.")

    with open(load_file, "rb") as f:
        data_dict = pickle.load(f)
    mha = FlashAttnV1(32, num_heads=4)
    mha = flash_attn_load_state_dict(mha, data_dict["model_state"])

    input_data = data_dict["input"].detach().numpy()
    output_data = data_dict["output"].detach().numpy()

    q_block_size = [1, 2, 4, 8, 16]
    kv_block_size = [1, 2, 4, 8, 16]

    for q_size, kv_size in itertools.product(q_block_size, kv_block_size):
        output = mha(input_data, q_block_size=q_size, kv_block_size=kv_size)
        match = np.allclose(output_data, output, atol=1e-7)
        result = "passed" if match else "failed"
        print(f"Check Flash Attn v1 forward, q: {q_size:2d}, kv: {kv_size:2d}, result: {result}")


if __name__ == "__main__":
    check_np_flash_attn_v1()
