"""Custom normalization layers."""
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn

from accelerator import get_accelerator
if get_accelerator().device_name() == "cuda":
    from vllm._C import ops


class RMSNorm(nn.Module):
    """Root mean square normalization.

    Computes x -> w * x / sqrt(E[x^2] + eps) where w is the learned weight.
    Refer to https://arxiv.org/abs/1910.07467
    """

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def _forward(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """PyTorch-native implementation equivalent to forward()."""
        orig_dtype = x.dtype
        x = x.to(torch.float32)
        if residual is not None:
            x = x + residual.to(torch.float32)
            residual = x.to(orig_dtype)

        variance = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.variance_epsilon)
        x = x.to(orig_dtype) * self.weight
        if residual is None:
            return x
        else:
            return x, residual

    def forward(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if residual is not None:
            if get_accelerator().device_name() == "xpu":
                x = torch.ops.torch_ipex.rms_norm(
                    x + residual,
                    [self.hidden_size],
                    self.weight.data,
                    self.variance_epsilon,
                )[0]
            else:
                ops.fused_add_rms_norm(
                    x,
                    residual,
                    self.weight.data,
                    self.variance_epsilon,
                )
            return x, residual
        out = torch.empty_like(x)
        if get_accelerator().device_name() == "xpu":
            out = torch.ops.torch_ipex.rms_norm(
                x,
                [self.hidden_size],
                self.weight.data,
                self.variance_epsilon,
            )[0]
        else:
            ops.rms_norm(
                out,
                x,
                self.weight.data,
                self.variance_epsilon,
            )
        return out
