#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np

import torch
from torch import nn


class RoPE(nn.Module):

    def __init__(
        self,
        dim: int,
        base: int = 10000,
        max_position_embeddings: int = 2048,
    ):
        super().__init__()
        self.dim = dim
        self.base = base
        self.max_position_embeddings = max_position_embeddings
        inv_freq = self.inverse_freq()
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def inverse_freq(self):
        idx = torch.arange(0, self.dim, 2, dtype=torch.int64).float()
        inv_freq = 1 / (self.base ** (idx / self.dim))
        return inv_freq

    @torch.no_grad()
    def forward(self, device: torch.device, position_ids: torch.Tensor):
        assert position_ids.max() < self.max_position_embeddings
        # force float32 accoring to https://github.com/huggingface/transformers/pull/29285
        with torch.autocast(device_type=device.type, enabled=False):
            freq = torch.einsum("j, bi->bij", self.inv_freq.float(), position_ids.float())
            embed = torch.cat([freq, freq], dim=-1)
            sin, cos = embed.sin(), embed.cos()
        return cos.to(device), sin.to(device)


def rotate_half(x):
    split_idx = x.shape[-1] // 2
    x1, x2 = x[..., :split_idx], x[..., split_idx:]
    assert x1.shape == x2.shape
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary_pos_emb(q, k, sin_pos, cos_pos):
    rope_q = q * cos_pos + rotate_half(q) * sin_pos
    rope_k = k * cos_pos + rotate_half(k) * sin_pos
    return rope_q, rope_k


def upper_bound(d, m) -> float:
    f = lambda x: 10000**(-2 * x / d)  # noqa

    return np.sum([
        np.linalg.norm(
            np.sum([np.exp(1j * m * f(i)) for i in range(j+1)])
        ) for j in range(d//2)
    ]) / (d/2)


def plot_rope_relative_distance(dim=128, max_value=256):
    m_values = np.arange(0, max_value + 1)
    f_values = [upper_bound(dim, m) for m in m_values]

    plt.plot(m_values, f_values)
    plt.xlabel('relative distance')
    plt.ylabel('relative upper bound')
    plt.show()


def test_rope_qk():
    seq, dim = 1024, 512
    x = torch.rand(2, seq, dim)
    pos = torch.arange(seq).unsqueeze(0)
    model = RoPE(dim=dim)
    cos, sin = model(x.device, pos)
    q, k = torch.rand(1, seq, dim), torch.rand(1, seq, dim)
    rope_q, rope_k = apply_rotary_pos_emb(q, k, sin, cos)
    assert rope_q.shape == q.shape
    assert rope_k.shape == k.shape
    print("test_rope_qk passed.")


if __name__ == "__main__":
    test_rope_qk()
    # plot_rope_relative_distance()
