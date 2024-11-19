#!/usr/bin/env python3

# A simple implementation of the sequence parallel block in Megatron-LM
# Reference paper: https://arxiv.org/pdf/2205.05198
# Check figure 5 in the paper for the architecture of the sequence parallel block.

import numpy as np

from mpi import (  # isort:skip
    all_gather,
    auto_split,
    barrier,
    log_rank,
    reduce_scatter,
    init_env,
    mpi_frame,
)
from linear_mpi import ParallelFFN  # isort:skip
from mha_mpi import TensorParallelMHA, load_from_file  # isort:skip


class LayerNorm:

    def __init__(self, normalized_shape, epsilon: float = 1e-5):
        self.normalized_shape = normalized_shape
        self.epsilon = epsilon
        self.gamma = np.ones(normalized_shape)
        self.beta = np.zeros(normalized_shape)

    def __call__(self, x):
        mean = np.mean(x, axis=-1, keepdims=True)
        variance = np.var(x, axis=-1, keepdims=True)

        x_normalized = (x - mean) / np.sqrt(variance + self.epsilon)

        output = self.gamma * x_normalized + self.beta
        return output


class SeqParallelBlock:

    def __init__(self, dim: int, mha_module, ffn_w1, ffn_w2):
        self.dim = dim

        self.ln1 = LayerNorm(dim)
        self.ln2 = LayerNorm(dim)
        self.mha = mha_module
        self.ffn = ParallelFFN(ffn_w1, ffn_w2)

    def __call__(self, x):
        # TODO: fix all gather hang issue
        x = self.ln1(x)  # sequence parallel layernorm
        barrier()
        x = all_gather(x, axis=1)  # gather along sequence length
        x = self.mha(x)  # tensor parallel MHA

        x = reduce_scatter(x)
        x = self.ln2(x)  # sequence parallel layernorm
        return x


def megatron_sp_forward(rank, world_size, queue, signal_queue, pipe_pairs):
    init_env(rank, world_size, queue, signal_queue, pipe_pairs)

    dim = 32
    ffn_w1, ffn_w2 = np.random.randn(dim, dim), np.random.randn(dim, dim)
    q_proj, k_proj, v_proj, out_proj, input_data, _ = load_from_file()

    mha = TensorParallelMHA(
        q_proj=q_proj,
        k_proj=k_proj,
        v_proj=v_proj,
        out_proj=out_proj,
        num_heads=4,
    )
    model = SeqParallelBlock(dim=32, mha_module=mha, ffn_w1=ffn_w1, ffn_w2=ffn_w2)

    input_data = auto_split(input_data, axis=1)  # split along sequence length
    out = model(input_data)
    log_rank(f"Input shape: {input_data.shape}, Output shape: {out.shape}")


def run_megatron_sp():
    for world_size in [1, 2, 4]:
        print(f"\nRun megatron sp with world size {world_size}")
        mpi_frame(megatron_sp_forward, world_size=world_size)
    print("\nMegatron SP passed!")


if __name__ == "__main__":
    run_megatron_sp()
