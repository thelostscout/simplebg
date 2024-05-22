import torch


def reconstruction(metrics, **kwargs):
    return torch.nn.functional.mse_loss(metrics.x, metrics.x1, reduction="none")
