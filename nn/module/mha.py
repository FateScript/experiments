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


def split_heads(x, num_heads: int):
    batch, seq, _ = x.shape
    # split model_dim to (num_heads, tensor_dim)
    x = x.reshape(batch, seq, num_heads, -1)
    # transpose to (batch, num_heads, seq_len, tensor_dim)
    x = x.transpose(1, 2)
    return x


class MultiHeadAttn(nn.Module):
    def __init__(self, dim: int, num_heads: int, use_kv_cache: bool = False):
        # NOTE: To visualize kv cache algorithm, try the following gif:
        # https://miro.medium.com/v2/resize:fit:1400/format:webp/1*uyuyOW1VBqmF5Gtv225XHQ.gif
        assert dim % num_heads == 0, f"dim must be divisible by num_heads, got dim {dim} and head {num_heads}"  # noqa
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

        h = self.num_heads
        q, k, v = split_heads(q, h), split_heads(k, h), split_heads(v, h)
        out = self.attn(q, k, v, mask=mask)
        out = out.transpose(1, 2).reshape(batch, seq, dim)
        return self.out_proj(out)


class MultiQueryAttn(nn.Module):
    def __init__(self, dim: int, num_heads: int, use_kv_cache: bool = False):
        assert dim % num_heads == 0, f"dim must be divisible by num_heads, got dim {dim} and head {num_heads}"  # noqa
        super().__init__()
        head_dim = dim // num_heads
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, head_dim, bias=False)
        self.v_proj = nn.Linear(dim, head_dim, bias=False)
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

        q = split_heads(q, num_heads=self.num_heads)
        k = k.unsqueeze(1).repeat(1, self.num_heads, 1, 1)
        v = v.unsqueeze(1).repeat(1, self.num_heads, 1, 1)

        out = self.attn(q, k, v, mask=mask)
        out = out.transpose(1, 2).reshape(batch, seq, dim)
        return self.out_proj(out)


class GroupQueryAttn(nn.Module):
    def __init__(
        self, dim: int, num_heads: int, num_groups: int, use_kv_cache: bool = False
    ):
        assert dim % num_heads == 0, f"dim must be divisible by num_heads, got dim {dim} and head {num_heads}"  # noqa
        assert num_heads % num_groups == 0, f"num_heads must be divisible by num_groups, got head {num_heads} and group {num_groups}"  # noqa

        super().__init__()
        head_dim = dim // num_heads
        group_dim = head_dim * num_groups

        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, group_dim, bias=False)
        self.v_proj = nn.Linear(dim, group_dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)

        self.attn = ScaledDotProductAttn()
        self.dim = dim
        self.num_heads = num_heads
        self.num_groups = num_groups
        self.repeat_size = num_heads // num_groups

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

        q = split_heads(q, self.num_heads)
        k, v = split_heads(k, self.num_groups), split_heads(v, self.num_groups)
        k = k.repeat(1, self.repeat_size, 1, 1)
        v = v.repeat(1, self.repeat_size, 1, 1)

        out = self.attn(q, k, v, mask=mask)
        out = out.transpose(1, 2).reshape(batch, seq, dim)
        return self.out_proj(out)


def save_mha_input_output(batch: int = 2, seq: int = 16, dim: int = 32, heads: int = 4):
    # save input, output and model state dict
    # used for numpy version MHA in experiments/se/mpi
    x = torch.randn(batch, seq, dim)
    model = MultiHeadAttn(dim, heads)
    out = model(x)
    model_state = model.state_dict()
    data_dict = {"input": x, "output": out, "model_state": model_state}
    save_file = "mha_input_output.pkl"
    with open(save_file, "wb") as f:
        pickle.dump(data_dict, f)
    print(f"Dump data to {save_file}")


def test_gqa_mha(batch: int = 2, seq: int = 16, dim: int = 32, heads: int = 4):
    x = torch.randn(batch, seq, dim)
    mha_model = MultiHeadAttn(dim, heads)
    gqa_model = GroupQueryAttn(dim, heads, num_groups=heads)
    gqa_model.load_state_dict(mha_model.state_dict())

    mha_out = mha_model(x)
    gqa_out = gqa_model(x)
    assert torch.allclose(mha_out, gqa_out, atol=1e-7), "Check failed"
    print("\nPassed GQA MHA test")


def test_gqa_mqa(batch: int = 2, seq: int = 16, dim: int = 32, heads: int = 4):
    x = torch.randn(batch, seq, dim)
    mqa_model = MultiQueryAttn(dim, heads)
    gqa_model = GroupQueryAttn(dim, heads, num_groups=1)
    gqa_model.load_state_dict(mqa_model.state_dict())

    mqa_out = mqa_model(x)
    gqa_out = gqa_model(x)
    assert torch.allclose(mqa_out, gqa_out, atol=1e-7), "Check failed"
    print("\nPassed GQA MQA test")


def check_kv_output_data(outputs, outputs_wo_kv):
    token_length = len(outputs)
    for idx in range(1, token_length + 1):
        data = torch.concat(outputs[:idx], dim=1)
        assert torch.allclose(data, outputs_wo_kv[idx - 1], atol=1e-7), "Check failed"
    print("\nPassed kv cache test")


def test_kv_cache(
    batch: int = 2,
    seq: int = 16,
    dim: int = 32,
    num_heads: int = 4,
    attn_type: str = "mha",
):
    assert attn_type in ["mha", "mqa", "gqa"]
    kwargs = {}
    if attn_type == "gqa":
        attn_class = GroupQueryAttn
        kwargs["num_groups"] = 2
    else:
        attn_class = MultiHeadAttn if attn_type == "mha" else MultiQueryAttn

    x = torch.randn(batch, seq, dim)
    x1, x2 = torch.randn(batch, 1, dim), torch.randn(batch, 1, dim)
    seq = [x, x1, x2]

    model = attn_class(dim, num_heads, use_kv_cache=True, **kwargs)

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
    model_wo_kv = attn_class(dim, num_heads, use_kv_cache=False, **kwargs)
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
    torch.manual_seed(3)
    save_mha_input_output()
    test_gqa_mha()
    test_gqa_mqa()

    for attn_type in ["mha", "mqa", "gqa"]:
        print(f"\nRun kv cache test with {attn_type}")
        test_kv_cache(attn_type=attn_type)
