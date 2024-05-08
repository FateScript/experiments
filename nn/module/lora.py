#!/usr/bin/env python3

import math
from typing import Optional
import torch
from torch import nn


class LoRA(nn.Module):

    def __init__(
        self,
        module: nn.Linear,
        rank: int,
        alpha: Optional[int] = None,
        dropout_prob: float = 0,
    ):
        super().__init__()
        out_dim, in_dim = module.weight.shape
        self.module = module
        self.lora_A = nn.Parameter(torch.empty(in_dim, rank))
        self.lora_B = nn.Parameter(torch.empty(rank, out_dim))
        self.r = rank
        self.alpha = alpha or self.r
        self.scale = self.alpha / self.r
        self.lora_dropout = lambda x: x
        if dropout_prob > 0:
            self.lora_dropout = nn.Dropout(p=dropout_prob)
        self.reset_parameters()

    def reset_parameters(self):
        # according to https://arxiv.org/pdf/2106.09685,
        # use a random Gaussian initialization for A and
        # zero for B, so ∆W = BA is zero at the beginning of training.
        # For a deeper thinking, see `ZeroInitLoRA` and `SwitchInitLoRA`
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))  # the same as nn.Linear.weight
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        out = self.module(x)
        lora_out = self.lora_dropout(x) @ self.lora_A @ self.lora_B
        return out + lora_out * self.scale

    def merge(self) -> nn.Parameter:
        with torch.no_grad():
            merged_weights = self.lora_A @ self.lora_B
            param = self.module.weight + merged_weights.T
        return param


class ZeroInitLoRA(LoRA):

    def reset_parameters(self):
        # use zero init for both lora weights A and B.
        # The lora_A/B will always be zero if you init them with zeros.
        nn.init.zeros_(self.lora_A)
        nn.init.zeros_(self.lora_B)


class SwithInitLoRA(LoRA):

    def reset_parameters(self):
        # switch init method of lora A/B
        # since `lora_out = input_feature @ lora_A @ lora_B`
        # if lora_A is set to all zeros, the input feature will have no effect
        # on training from the begining and it may cost more time for lora_B to converge.
        nn.init.zeros_(self.lora_A)
        nn.init.kaiming_uniform_(self.lora_B, a=math.sqrt(5))


def train(
    iters: int = 50,
    input_channel: int = 128,
    output_channel: int = 256,
    lora_type: str = "normal",
):
    assert lora_type in ("normal", "zero", "switch"), f"{lora_type} not supported"
    linear_module = nn.Linear(input_channel, output_channel, bias=False)
    linear_module.eval()

    LoRA_classs = LoRA
    if lora_type == "zero":
        LoRA_classs = ZeroInitLoRA
    elif lora_type == "switch":
        LoRA_classs = SwithInitLoRA

    lora_module = LoRA_classs(linear_module, rank=32)
    optimizer = torch.optim.AdamW(lora_module.parameters(), lr=1e-2, weight_decay=1e-1)

    for cur_iter in range(iters):
        data = torch.rand(2, input_channel)
        out = lora_module(data)
        loss = torch.abs(out.mean())
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        print(f"iter {cur_iter}: loss {loss.item(): .4f}")

    return lora_module


def test_init():
    input_channel, output_channel = 128, 256
    linear_module = nn.Linear(input_channel, output_channel)
    lora_module = LoRA(linear_module, rank=32)
    out = lora_module.merge()
    assert torch.equal(out, linear_module.weight)
    print("Pass init test!")


def test_merge():
    input_channel, output_channel = 128, 256
    inp = torch.rand(3, input_channel)
    lora_module = train(
        iters=50,
        input_channel=input_channel,
        output_channel=output_channel
    )
    with torch.no_grad():
        lora_module.eval()
        merge_weights = lora_module.merge()
        out = lora_module(inp)
        eq_out = inp @ merge_weights.T
        max_value = torch.abs(out - eq_out).max()
        assert max_value < 1e-6
    print("Pass merge test!")


def test_zero_lora():
    input_channel, output_channel = 128, 256
    lora_module = train(
        iters=50,
        input_channel=input_channel,
        output_channel=output_channel,
        lora_type="zero",
    )
    assert torch.equal(lora_module.lora_A, torch.zeros_like(lora_module.lora_A))
    assert torch.equal(lora_module.lora_B, torch.zeros_like(lora_module.lora_B))
    print("Zero Init version doesn't train as expected!")


if __name__ == "__main__":
    test_init()
    test_merge()
    test_zero_lora()
