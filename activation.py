import torch
from torch import nn
import torch.nn.functional as F

from context import get_context


def silu_and_mul(x: torch.Tensor) -> torch.Tensor:
    x, y = x.chunk(2, -1)
    return F.silu(x) * y


silu_and_mul_compiled = torch.compile(silu_and_mul)


class SiluAndMul(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        fn = silu_and_mul if get_context().canonical else silu_and_mul_compiled
        return fn(x)
