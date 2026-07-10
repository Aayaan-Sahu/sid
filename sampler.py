import torch
from torch import nn


class Sampler(nn.Module):
    @torch.compile
    def forward(self, logits: torch.Tensor):
        return torch.argmax(logits, dim=-1)