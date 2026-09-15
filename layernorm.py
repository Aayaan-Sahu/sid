import torch
from torch import nn

from context import get_context


def rms_forward(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    orig_dtype = x.dtype
    x = x.float()
    var = x.pow(2).mean(dim=-1, keepdim=True)
    x.mul_(torch.rsqrt(var + eps))
    x = x.to(orig_dtype).mul_(weight)
    return x


def add_rms_forward(x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float) -> tuple[torch.Tensor, torch.Tensor]:
    orig_dtype = x.dtype
    x = x.float().add_(residual.float())
    residual = x.to(orig_dtype)
    var = x.pow(2).mean(dim=-1, keepdim=True)
    x.mul_(torch.rsqrt(var + eps))
    x = x.to(orig_dtype).mul_(weight)
    return x, residual


rms_forward_compiled = torch.compile(rms_forward)
add_rms_forward_compiled = torch.compile(add_rms_forward)


class RMSNorm(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        canonical = get_context().canonical
        if residual is None:
            fn = rms_forward if canonical else rms_forward_compiled
            return fn(x, self.weight, self.eps)
        else:
            fn = add_rms_forward if canonical else add_rms_forward_compiled
            return fn(x, residual, self.weight, self.eps)
