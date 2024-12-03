#!/usr/bin/env python3

# paper: https://arxiv.org/pdf/2108.12409
# code: https://github.com/ofirpress/attention_with_linear_biases/

import math

import torch
from torch import nn

from mha import check_kv_output_data, split_heads  # local import  # isort:skip


def alibi_slopes(num_heads: int, alibi_max_bias: int = 8):
    # according to part 3 of paper, use 2 ** (-8 / n) as start and ratio
    # code reference:
    # https://github.com/ofirpress/attention_with_linear_biases/blob/4b92f28a005ead2567abe2359f633e73e08f3833/fairseq/models/transformer.py#L742-L752   # noqa
    start = alibi_max_bias / num_heads
    slopes = torch.arange(start, alibi_max_bias + start, step=start)
    slopes = torch.pow(2, -slopes)
    return slopes


def attn_linear_bias(
    seq_len: int,
    num_heads: int,
    alibi_max_bias: int = 8,
    use_kv_cache: bool = False,
):
    """Creates the ALiBi bias matrix."""
    # See figure-3 in paper
    slopes = alibi_slopes(num_heads, alibi_max_bias)  # head-specific slopes

    if use_kv_cache:
        distance = torch.arange(-seq_len + 1, 1).view(1, seq_len)
    else:
        distance = torch.arange(seq_len).view(1, -1) - torch.arange(seq_len).view(-1, 1)

    bias = distance.unsqueeze(0) * slopes.view(num_heads, 1, 1)  # add head dim for distance
    # the up-right part will be masked, so ignore the part
    return bias.unsqueeze(0)  # add batch dim


class Attn(nn.Module):

    def __init__(self, return_attn=False):
        super().__init__()
        self.return_attn = return_attn
        self.pass_first_kv_cache = False

    def forward(self, q, k, v, mask=None, use_kv_cache: bool = False):
        *_, num_heads, seq_len, dim = k.size()  # (batch, head, length, dim)

        if use_kv_cache and not self.pass_first_kv_cache:
            # use normal alibi when use_kv_cache the first time
            self.pass_first_kv_cache = True
            use_kv_cache = False

        scale = math.sqrt(dim)
        alibi = attn_linear_bias(seq_len, num_heads, use_kv_cache=use_kv_cache)

        mat = (q @ k.transpose(-1, -2)) / scale + alibi  # accord to paper, alibi is not scaled
        if mask is not None:
            mat = mat.masked_fill(mask == 0, -1e6)
        mat = torch.softmax(mat, dim=-1)

        v = mat @ v
        if self.return_attn:
            return v, mat
        return v


class MultiHeadAttnAlibi(nn.Module):
    def __init__(self, dim: int, num_heads: int, use_kv_cache: bool = False):
        # NOTE: To visualize kv cache algorithm, try the following gif:
        # https://miro.medium.com/v2/resize:fit:1400/format:webp/1*uyuyOW1VBqmF5Gtv225XHQ.gif
        assert dim % num_heads == 0, f"dim must be divisible by num_heads, got dim {dim} and head {num_heads}"  # noqa
        super().__init__()
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)

        self.attn = Attn()  # The only difference between MultiHeadAttn and MultiHeadAttnAlibi
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
        # The only difference between MultiHeadAttn and MultiHeadAttnAlibi
        out = self.attn(q, k, v, mask=mask, use_kv_cache=self.use_kv_cache)
        out = out.transpose(1, 2).reshape(batch, seq, dim)
        return self.out_proj(out)


def test_kv_cache(
    batch: int = 2,
    seq: int = 16,
    dim: int = 32,
    num_heads: int = 4,
):
    x = torch.randn(batch, seq, dim)
    x1, x2 = torch.randn(batch, 1, dim), torch.randn(batch, 1, dim)
    seq = [x, x1, x2]

    model = MultiHeadAttnAlibi(dim, num_heads, use_kv_cache=True)

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
    model_wo_kv = MultiHeadAttnAlibi(dim, num_heads, use_kv_cache=False)
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
    test_kv_cache()
