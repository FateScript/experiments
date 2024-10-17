#!/usr/bin/env python3

import math
import pickle

import torch
from torch import nn


class ScaledDotProductAttn(nn.Module):

    def __init__(self, return_attn=False):
        super().__init__()
        self.return_attn = return_attn

    def forward(self, q, k, v, mask=None):
        *_, dim = q.size()  # usually (batch, seq_len, dim) or (batch, head, length, dim)
        scale = math.sqrt(dim)
        mat = (q @ k.transpose(-1, -2)) / scale
        if mask:
            mat = mat.masked_fill(mask == 0, -1e6)
        mat = torch.softmax(mat, dim=-1)

        v = mat @ v
        if self.return_attn:
            return v, mat
        return v


class MultiHeadAttn(nn.Module):

    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)

        self.attn = ScaledDotProductAttn()
        self.dim = dim
        self.num_heads = num_heads

    def forward(self, x, mask=None):
        batch, seq, dim = x.shape
        q, k, v = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        q, k, v = self.split_heads(q), self.split_heads(k), self.split_heads(v)
        out = self.attn(q, k, v, mask=mask)
        out = out.transpose(1, 2).reshape(batch, seq, dim)
        return self.out_proj(out)

    def split_heads(self, x):
        batch, seq, dim = x.shape
        # split model_dim to (num_heads, tensor_dim)
        x = x.reshape(batch, seq, self.num_heads, -1)
        # transpose to (batch, num_heads, seq_len, tensor_dim)
        x = x.transpose(1, 2)
        return x


def save_input_output():
    # save input, output and model state dict
    # used for numpy version MHA in experiments/se/mpi
    batch, seq, dim = 2, 16, 32
    x = torch.randn(batch, seq, dim)
    model = MultiHeadAttn(dim, 4)
    out = model(x)
    model_state = model.state_dict()
    data_dict = {"input": x, "output": out, "model_state": model_state}
    save_file = "mha_input_output.pkl"
    with open(save_file, "wb") as f:
        pickle.dump(data_dict, f)
    print(f"Dump data to {save_file}")


if __name__ == "__main__":
    save_input_output()
