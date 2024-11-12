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
        if mask is not None:
            mat = mat.masked_fill(mask == 0, -1e6)
        mat = torch.softmax(mat, dim=-1)

        v = mat @ v
        if self.return_attn:
            return v, mat
        return v


class MultiHeadAttn(nn.Module):

    def __init__(self, dim: int, num_heads: int, use_kv_cache: bool = False):
        # NOTE: To visualize kv cache algorithm, try the following gif:
        # https://miro.medium.com/v2/resize:fit:1400/format:webp/1*uyuyOW1VBqmF5Gtv225XHQ.gif
        super().__init__()
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)

        self.attn = ScaledDotProductAttn()
        self.dim = dim
        self.num_heads = num_heads

        self.use_kv_cache = use_kv_cache
        if self.use_kv_cache:
            self.k_cache = []
            self.v_cache = []

    def forward(self, x, mask=None):
        batch, seq, dim = x.shape
        q, k, v = self.q_proj(x), self.k_proj(x), self.v_proj(x)

        if self.use_kv_cache:
            prev_k, prev_v = k, v
            k = torch.concat([*self.k_cache, prev_k], dim=1)
            v = torch.concat([*self.v_cache, prev_v], dim=1)
            self.k_cache.append(prev_k)
            self.v_cache.append(prev_v)

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


def check_kv_output_data(outputs, outputs_wo_kv):
    token_length = len(outputs)
    for idx in range(1, token_length + 1):
        data = torch.concat(outputs[:idx], dim=1)
        assert torch.allclose(data, outputs_wo_kv[idx-1]), "Output not match"
    print("\nPassed kv cache test")


def test_kv_cache():
    batch, seq, dim = 2, 16, 32
    x = torch.randn(batch, seq, dim)
    x1, x2 = torch.randn(batch, 1, dim), torch.randn(batch, 1, dim)
    seq = [x, x1, x2]
    model = MultiHeadAttn(dim, 4, use_kv_cache=True)

    print("Inference with kv cache")
    outputs = []
    for idx in range(len(seq)):
        data, mask = seq[idx], None
        if idx == 0:
            seq_len = data.size(1)
            mask = torch.tril(torch.ones(seq_len, seq_len))
        with torch.no_grad():
            out = model(data, mask=mask)
        outputs.append(out)
        print(f"Outpt shape with kv: {out.shape}")

    print("\nInference without kv cache")
    model_wo_kv = MultiHeadAttn(dim, 4, use_kv_cache=False)
    model_wo_kv.load_state_dict(model.state_dict())
    outputs_wo_kv = []
    for idx in range(1, len(seq) + 1):
        data = torch.concat(seq[:idx], dim=1)
        seq_len = data.size(1)
        mask = torch.tril(torch.ones(seq_len, seq_len))
        with torch.no_grad():
            out = model_wo_kv(data, mask=mask)
        print(f"Shape without kv: {out.shape}")
        outputs_wo_kv.append(out)

    check_kv_output_data(outputs, outputs_wo_kv)


if __name__ == "__main__":
    torch.manual_seed(42)
    # save_input_output()
    test_kv_cache()
