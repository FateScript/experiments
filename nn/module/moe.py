#!/usr/bin/env python3

import torch
from torch import nn
import torch.nn.functional as F


class MoE(nn.Module):

    def __init__(
        self,
        hidden_dim: int,
        num_experts: int,
        topk: int,
        num_shared_experts: int,
        norm_probs: bool = True,
        seq_level: bool = False,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.num_shared_experts = num_shared_experts
        self.topk = topk
        self.norm_probs = norm_probs
        self.seq_level = seq_level

        self.routers = nn.Linear(hidden_dim, self.num_experts, bias=False)
        self.shared_experts = nn.Linear(hidden_dim, hidden_dim * num_shared_experts, bias=False)
        self.experts = [
            nn.Linear(hidden_dim, hidden_dim, bias=False)
            for _ in range(self.num_experts)
        ]

    def forward(self, feat: torch.Tensor):
        batch, seq, dim = feat.shape
        flat_feat = feat.reshape(-1, dim)
        router_feat = self.routers(flat_feat)

        # shared expert feature
        shared_feats = self.shared_experts(flat_feat)
        shared_feats = shared_feats.reshape(-1, self.num_shared_experts, dim).sum(dim=-2)

        topk_idx, weights = self.gate_topk(router_feat)

        aux_loss = self.balance_loss(router_feat, topk_idx, batch, seq)
        flat_topk = topk_idx.flatten()
        if self.training:
            flat_expert_feats = flat_feat.repeat_interleave(self.topk, dim=0)
            expert_feat = torch.zeros_like(flat_expert_feats)

            for expert_id in range(self.num_experts):
                select_token = flat_topk == expert_id
                expert_feat[select_token] = self.experts[expert_id](flat_expert_feats[select_token])

            weighted_feat = (weights.reshape(-1, 1) * expert_feat).reshape(-1, self.topk, dim).sum(dim=1)
        else:
            weighted_feat = self.inference(flat_feat, flat_topk, weights)

        out_feat = shared_feats + weighted_feat
        return out_feat.reshape(batch, seq, dim), aux_loss

    def gate_topk(self, router_feat):
        if self.norm_probs:
            topk_prob, topk_idx = torch.topk(router_feat, k=self.topk, sorted=False)
            weights = torch.softmax(topk_prob, dim=-1)
        else:
            router_prob = torch.softmax(router_feat, dim=-1)
            weights, topk_idx = torch.topk(router_prob, k=self.topk, sorted=False)
        return topk_idx, weights

    @torch.no_grad()
    def inference(
        self, feat: torch.Tensor, topk_idx: torch.Tensor, weights: torch.Tensor
    ):
        moe_feat = torch.zeros_like(feat)
        weights = weights.flatten()
        topk_id_mapping = topk_idx.argsort()
        re_weights = weights.flatten()[topk_id_mapping]
        topk_id_mapping = topk_id_mapping // self.topk  # div topk to get expert idx
        expert_end_idx = topk_idx.bincount().cumsum(dim=0).cpu().numpy()
        for idx, end_idx in enumerate(expert_end_idx):
            start_idx = 0 if idx == 0 else expert_end_idx[idx - 1]
            if start_idx == end_idx:  # no expert assigned
                continue
            mapping, w = topk_id_mapping[start_idx:end_idx], re_weights[start_idx:end_idx]
            expert_feat = self.experts[idx](feat[mapping]) * w.unsqueeze(-1)
            moe_feat[mapping] += expert_feat
        return moe_feat

    def balance_loss(self, router_feat, topk_idx, batch, seq):
        scores = torch.softmax(router_feat, dim=-1)

        aux_loss = None
        if self.training:
            if self.seq_level:
                topk_idx = topk_idx.reshape(batch, -1)
                scores = scores.view(batch, seq, -1)
                bin_count = torch.zeros(batch, self.num_experts)
                bin_count.scatter_add_(1, topk_idx, torch.ones(batch, seq * self.topk))
                ce = bin_count / (seq * self.topk / self.num_experts)
                aux_loss = (ce * scores.mean(dim=1)).sum(dim=1).mean()
            else:
                bin_count = F.one_hot(topk_idx.flatten(), num_classes=self.num_experts)
                fi = bin_count.sum(dim=0) / (batch * seq)
                pi = scores.mean(dim=0)
                aux_loss = (pi * fi).sum()
        return aux_loss


def test_eval_train_match():
    hidden, routers, topk, shared = 16, 8, 2, 2
    batch, seq = 2, 16
    model = MoE(hidden, routers, topk, shared)
    model.eval()
    feat = torch.rand((batch, seq, hidden))
    feat.requires_grad_()
    out, _ = model(feat)
    model.train()
    out2, _ = model(feat)
    assert out.shape == feat.shape
    assert torch.allclose(out, out2)


def test_balance_loss():
    hidden, routers, topk, shared = 16, 8, 2, 2
    batch, seq = 2, 16
    model = MoE(hidden, routers, topk, shared, seq_level=False)
    feat = torch.rand((batch, seq, hidden))
    feat.requires_grad_()
    out, loss = model(feat)

    model = MoE(hidden, routers, topk, shared, seq_level=True)
    out2, loss2 = model(feat)
    breakpoint()
    pass


if __name__ == "__main__":
    test_eval_train_match()
    test_balance_loss()
    print("Passed!")
