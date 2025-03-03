#!/usr/bin/env python3

# To check the difference between loss scale and lr scale
# NOTE that: update_grad = grad + wd * param
# reference code:
# torch:
# https://github.com/pytorch/pytorch/blob/ab81ca5053440074dc7d8c46ae4775f62f662394/torch/optim/sgd.py#L345
# megengine:
# https://github.com/MegEngine/MegEngine/blob/47952c075d868665e1116214bea760d786144081/imperative/python/megengine/optimizer/sgd.py#L85

import torch
from torch import optim


def make_optimizer(
    model,
    optimizer_name: str,
    lr: float = 0.1,
    wd: float = 0.1,
):
    if optimizer_name == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    elif optimizer_name == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=wd)
    elif optimizer_name == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    else:
        raise ValueError(f'Unknown optimizer: {optimizer_name}')
    return optimizer


def train_step(model, optimizer, data, label, loss_scale=1.0):
    loss_fn = torch.nn.MSELoss()
    optimizer.zero_grad()
    y_pred = model(data)
    # loss = loss_fn(y_pred, label) * loss_scale
    loss = y_pred.sum() * loss_scale
    loss.backward()
    # print(model.weight.grad)
    optimizer.step()
    return model, loss.item()


def check():
    torch.manual_seed(42)
    optimizer_name = ["sgd", "adam", "adamw"]
    lr, wd, scalar = 1, 0.1, 10
    wd_choices = [0.0, 0.1]
    for name in optimizer_name:
        for wd in wd_choices:
            model = torch.nn.Linear(10, 10, bias=False)
            weights = model.weight.data.clone()
            x = torch.randn(2, 10)
            y = torch.randn(2, 10)
            optimizer = make_optimizer(model, name, lr=lr, wd=wd)

            # loss scale
            model, _ = train_step(model, optimizer, x, y, loss_scale=scalar)
            loss_params = model.weight.data

            # lr scale
            model2 = torch.nn.Linear(10, 10, bias=False)
            model2.weight.data = weights.clone()
            model2.weight.requires_grad_(True)
            scale_optim = make_optimizer(model2, name, lr=lr * scalar, wd=wd)
            model2, _ = train_step(model2, scale_optim, x, y, loss_scale=1.0)
            lr_params = model2.weight.data

            equal = torch.allclose(loss_params, lr_params, atol=1e-7)
            print(f"{name:5s} with weight decay {wd}: Loss scale == LR scale: {equal}")
        print()


if __name__ == "__main__":
    # only SGD with weight decay 0.0 is True
    check()
