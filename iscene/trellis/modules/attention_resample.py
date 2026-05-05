from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

try:
    import flash_attn
except ImportError:  # pragma: no cover - flash-attn is optional
    flash_attn = None

__all__ = ["AttentionResample"]


class AttentionResample(nn.Module):
    """Resample a variable-length token sequence to a fixed target length."""

    def __init__(
        self,
        d_model: int = 1024,
        n_target: int = 4096,
        *,
        n_heads: int = 16,
        use_flash: bool = True,
    ) -> None:
        super().__init__()

        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_target = n_target
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = self.head_dim ** -0.5

        self.latent = nn.Parameter(torch.randn(n_target, d_model))
        self.to_kv = nn.Linear(d_model, 2 * d_model, bias=False)
        self._flash_available = use_flash and flash_attn is not None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return a tensor with shape (B, n_target, d_model)."""
        batch_size, _, dim = x.shape
        assert dim == self.d_model, f"Expected input dim {self.d_model}, got {dim}"

        q = self.latent.unsqueeze(0).expand(batch_size, -1, -1)
        k, v = self.to_kv(x).chunk(2, dim=-1)

        if self._flash_available:
            return self._forward_flash(q, k, v)
        return self._forward_torch(q, k, v)

    def _forward_torch(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        batch_size = q.size(0)
        q = q.view(batch_size, self.n_target, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        weights = torch.softmax(attn, dim=-1, dtype=attn.dtype)
        out = torch.matmul(weights, v)
        return out.transpose(1, 2).contiguous().view(batch_size, self.n_target, self.d_model)

    def _forward_flash(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        batch_size = q.size(0)
        q = q.view(batch_size, self.n_target, self.n_heads, self.head_dim).contiguous()
        k = k.view(batch_size, -1, self.n_heads, self.head_dim).contiguous()
        v = v.view(batch_size, -1, self.n_heads, self.head_dim).contiguous()

        assert flash_attn is not None
        out = flash_attn.flash_attn_func(
            q,
            k,
            v,  # type: ignore[arg-type]
            causal=False,
            softmax_scale=self.scale,
        )
        return out.reshape(batch_size, self.n_target, self.d_model)
