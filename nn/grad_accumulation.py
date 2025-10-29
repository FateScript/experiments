import itertools

import numpy as np
import torch
import torch.nn as nn


def create_model():
    model = nn.Linear(3, 1, bias=False)
    with torch.no_grad():
        model.weight.data = torch.tensor([[1.0, 2.0, 3.0]])
    return model


def generate_data(total_samples=4):
    X = torch.randn(total_samples, 3)
    y = torch.randn(total_samples, 1)
    return X, y


def train_model(model, data, label, loss_func, accum_steps=1):
    step_batch_size = data.shape[0] // accum_steps
    total_loss = 0.0

    for i in range(accum_steps):
        batch_slice = slice(i*step_batch_size, (i+1)*step_batch_size)
        batch_data = data[batch_slice]
        batch_label = label[batch_slice]

        output = model(batch_data)
        loss = loss_func(output, batch_label)
        loss.backward()

        total_loss += loss.item()
    return model, total_loss


def run_experiments():
    data, label = generate_data(4)

    models = []
    for idx, (step, reduction) in enumerate(itertools.product([2, 1], ["mean", "sum"]), start=1):
        print("\n" + "=" * 80)
        experiment_title = f"实验{idx}: 梯度累积 {step} 次, 使用 loss.{reduction}()"
        print(experiment_title)

        model = create_model()
        loss_fn = nn.MSELoss(reduction=reduction)
        model, total_loss = train_model(model, data, label, loss_fn, accum_steps=step)

        print(f"\n梯度累积后的总梯度: {model.weight.grad.data}")
        if reduction == "mean":
            total_loss /= step
        print(f"loss: {total_loss:.6f}")
        models.append(model)

    return models


def main():
    torch.manual_seed(42)
    np.random.seed(42)

    model1, model2, model3, model4 = run_experiments()

    print("\n" + "-" * 80)
    print("Summary of Results")
    print(f"实验1 step=2, mean): {model1.weight.grad.data}")
    print(f"实验2 step=2, sum):  {model2.weight.grad.data}")
    print(f"实验3 step=1, mean): {model3.weight.grad.data}")
    print(f"实验4 step=1, sum):  {model4.weight.grad.data}")

    print(f"实验1的梯度是实验3的{(model1.weight.grad / model3.weight.grad).mean().item():.1f}倍")


if __name__ == "__main__":
    main()
