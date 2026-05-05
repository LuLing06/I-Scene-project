from typing import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from .full_attn import scaled_dot_product_attention
from einops import rearrange

class MultiHeadRMSNorm(nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(heads, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (F.normalize(x.float(), dim = -1) * self.gamma * self.scale).to(x.dtype)


class RotaryPositionEmbedder(nn.Module):
    def __init__(self, hidden_size: int, in_channels: int = 3):
        super().__init__()
        assert hidden_size % 2 == 0, "Hidden size must be divisible by 2"
        self.hidden_size = hidden_size
        self.in_channels = in_channels
        self.freq_dim = hidden_size // in_channels // 2
        self.freqs = torch.arange(self.freq_dim, dtype=torch.float32) / self.freq_dim
        self.freqs = 1.0 / (10000 ** self.freqs)

    def _get_phases(self, indices: torch.Tensor) -> torch.Tensor:
        self.freqs = self.freqs.to(indices.device)
        phases = torch.outer(indices, self.freqs)
        phases = torch.polar(torch.ones_like(phases), phases)
        return phases

    def _rotary_embedding(self, x: torch.Tensor, phases: torch.Tensor) -> torch.Tensor:
        x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
        x_rotated = x_complex * phases
        x_embed = torch.view_as_real(x_rotated).reshape(*x_rotated.shape[:-1], -1).to(x.dtype)
        return x_embed

    def forward(self, q: torch.Tensor, k: torch.Tensor, indices: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            q (sp.SparseTensor): [..., N, D] tensor of queries
            k (sp.SparseTensor): [..., N, D] tensor of keys
            indices (torch.Tensor): [..., N, C] tensor of spatial positions
        """
        if indices is None:
            indices = torch.arange(q.shape[-2], device=q.device)
            if len(q.shape) > 2:
                indices = indices.unsqueeze(0).expand(q.shape[:-2] + (-1,))

        phases = self._get_phases(indices.reshape(-1)).reshape(*indices.shape[:-1], -1)
        if phases.shape[1] < self.hidden_size // 2:
            phases = torch.cat([phases, torch.polar(
                torch.ones(*phases.shape[:-1], self.hidden_size // 2 - phases.shape[1], device=phases.device),
                torch.zeros(*phases.shape[:-1], self.hidden_size // 2 - phases.shape[1], device=phases.device)
            )], dim=-1)
        q_embed = self._rotary_embedding(q, phases)
        k_embed = self._rotary_embedding(k, phases)
        return q_embed, k_embed


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        channels: int,
        num_heads: int,
        ctx_channels: Optional[int]=None,
        type: Literal["self", "cross"] = "self",
        attn_mode: Literal["full", "windowed"] = "full",
        window_size: Optional[int] = None,
        shift_window: Optional[Tuple[int, int, int]] = None,
        qkv_bias: bool = True,
        use_rope: bool = False,
        qk_rms_norm: bool = False,
    ):
        super().__init__()
        assert channels % num_heads == 0
        assert type in ["self", "cross"], f"Invalid attention type: {type}"
        assert attn_mode in ["full", "windowed"], f"Invalid attention mode: {attn_mode}"
        assert type == "self" or attn_mode == "full", "Cross-attention only supports full attention"

        if attn_mode == "windowed":
            raise NotImplementedError("Windowed attention is not yet implemented")

        self.channels = channels
        self.head_dim = channels // num_heads
        self.ctx_channels = ctx_channels if ctx_channels is not None else channels
        self.num_heads = num_heads
        self._type = type
        self.attn_mode = attn_mode
        self.window_size = window_size
        self.shift_window = shift_window
        self.use_rope = use_rope
        self.qk_rms_norm = qk_rms_norm

        if self._type == "self":
            self.to_qkv = nn.Linear(channels, channels * 3, bias=qkv_bias)
        else:
            self.to_q = nn.Linear(channels, channels, bias=qkv_bias)
            self.to_kv = nn.Linear(self.ctx_channels, channels * 2, bias=qkv_bias)

        if self.qk_rms_norm:
            self.q_rms_norm = MultiHeadRMSNorm(self.head_dim, num_heads)
            self.k_rms_norm = MultiHeadRMSNorm(self.head_dim, num_heads)

        self.to_out = nn.Linear(channels, channels)

        if use_rope:
            self.rope = RotaryPositionEmbedder(channels)
        self.use_positional_encoding = False

    def initialize_positional_encoding(self, num_external_sources: int = 2, enable_gate: bool = True, enable_k_bias: bool = False, k_bias_scale: float = 0.1):
        self.use_positional_encoding = True
        # Controls for optional mechanisms
        self.enable_ext_gate = bool(enable_gate)
        self.enable_ext_k_bias = bool(enable_k_bias)
        self.ext_k_bias_scale = float(k_bias_scale)

        # K-gate for external keys only (values unchanged)
        if self.enable_ext_gate:
            self.ext_gate = nn.Parameter(torch.full((num_external_sources, self.num_heads,), 0.0))

        # Per-source, per-head K additive bias vector (bounded via tanh during application)
        if self.enable_ext_k_bias:
            self.k_type_bias = nn.Parameter(torch.zeros(num_external_sources, self.num_heads, self.head_dim))


    def forward(self, x: torch.Tensor, context: Optional[torch.Tensor] = None, indices: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, L, C = x.shape
        if self._type == "self":
            qkv = self.to_qkv(x)
            qkv = qkv.reshape(B, L, 3, self.num_heads, -1)
            if self.use_rope:
                q, k, v = qkv.unbind(dim=2)
                q, k = self.rope(q, k, indices)
                qkv = torch.stack([q, k, v], dim=2)
            if self.attn_mode == "full":
                if self.qk_rms_norm:
                    q, k, v = qkv.unbind(dim=2)
                    q = self.q_rms_norm(q)
                    k = self.k_rms_norm(k)
                    h = scaled_dot_product_attention(q, k, v)
                else:
                    h = scaled_dot_product_attention(qkv)
            elif self.attn_mode == "windowed":
                raise NotImplementedError("Windowed attention is not yet implemented")
        else:
            Lkv = context.shape[1]
            q = self.to_q(x)
            kv = self.to_kv(context)
            q = q.reshape(B, L, self.num_heads, -1)
            kv = kv.reshape(B, Lkv, 2, self.num_heads, -1)
            if self.qk_rms_norm:
                q = self.q_rms_norm(q)
                k, v = kv.unbind(dim=2)
                k = self.k_rms_norm(k)
                h = scaled_dot_product_attention(q, k, v)
            else:
                h = scaled_dot_product_attention(q, kv)
        h = h.reshape(B, L, -1)
        h = self.to_out(h)
        return h

    def mi_attention(self, x: torch.Tensor, num_instances: int, indices: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Multi-instance self-attention.
        q stays (B_total, L, ...).
        k, v are concatenated across instances (N) -> (B, N*L, ...), then expanded to (B*N, N*L, ...).
        """
        B_total, L, C = x.shape

        # 1. QKV projection
        qkv = self.to_qkv(x).reshape(B_total, L, 3, self.num_heads, -1)
        q, k, v = qkv.unbind(dim=2)

        # 2. RoPE
        if self.use_rope:
            q, k = self.rope(q, k, indices)

        if self.qk_rms_norm:
            q = self.q_rms_norm(q)
            k = self.k_rms_norm(k)

        # q: (B*N, L, H, D)

        # 3. Prepare K, V: merge instances in scene, then broadcast to each instance
        # (B*N, L, H, D) -> (B, N*L, H, D)
        k_scene = rearrange(k, '(b n) l h d -> b (n l) h d', n=num_instances)
        v_scene = rearrange(v, '(b n) l h d -> b (n l) h d', n=num_instances)

        # Expand to (B*N, N*L, H, D)
        # We want each of the N instances in batch b to see the same k_scene[b]
        # k_scene: (B, 1, NL, H, D) -> expand -> (B, N, NL, H, D) -> reshape -> (BN, NL, H, D)
        k_all = k_scene.unsqueeze(1).expand(-1, num_instances, -1, -1, -1)
        k_all = rearrange(k_all, 'b n nl h d -> (b n) nl h d')

        v_all = v_scene.unsqueeze(1).expand(-1, num_instances, -1, -1, -1)
        v_all = rearrange(v_all, 'b n nl h d -> (b n) nl h d')

        # 4. Attention
        # q: (BN, L, H, D)
        # k_all: (BN, NL, H, D)
        # out: (BN, L, H, D)
        h = scaled_dot_product_attention(q, k_all, v_all)

        # 6. Output projection
        h = h.reshape(B_total, L, -1)
        h = self.to_out(h)
        return h

    def scene_context_attn(self, x: torch.Tensor, context: torch.Tensor, num_instances=3, indices: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, L, C = x.shape

        # Project to QKV and apply rotary/QK RMS-norm as configured
        qkv = self.to_qkv(x).reshape(B, L, 3, self.num_heads, -1)
        q, k, v = qkv.unbind(dim=2)
        if self.use_rope:
            q, k = self.rope(q, k, indices)
        if self.qk_rms_norm:
            q = self.q_rms_norm(q)
            k = self.k_rms_norm(k)

        # Reshape into pairs: (bp, num_instances, L, H, C)
        qp = rearrange(q, '(bp ni) L h c -> bp ni L h c', ni=num_instances)
        kp = rearrange(k, '(bp ni) L h c -> bp ni L h c', ni=num_instances)
        vp = rearrange(v, '(bp ni) L h c -> bp ni L h c', ni=num_instances)

        output_list =[]
        ext_k_list = []
        for ins_idx in range(1, num_instances):
            k_j = kp[:, ins_idx]  # (bp, L, H, C)

            if self.use_positional_encoding:
                # pick a source id for this external (share or per-instance)
                # share: src_id = 0     # if you only defined one external source
                src_id = ins_idx - 1

                if getattr(self, 'enable_ext_k_bias', False):
                    bias = torch.tanh(self.k_type_bias[src_id])[None, None, :, :].to(dtype=k_j.dtype, device=k_j.device)
                    k_j = k_j + self.ext_k_bias_scale * bias

                if getattr(self, 'enable_ext_gate', False):
                    alpha = torch.sigmoid(self.ext_gate[src_id])[None, None, :, None].to(dtype=k_j.dtype, device=k_j.device)
                    k_j = k_j * alpha

            ext_k_list.append(k_j)

        k_full = torch.cat([kp[:, 0]] + ext_k_list, dim=1)  # (bp, num_instances * L, H, C)
        v_full = torch.cat([vp[:, i] for i in range(num_instances)], dim=1)
        out_inst = scaled_dot_product_attention(qp[:, 0], k_full, v_full)
        output_list.append(out_inst)

        # num_instance > 1 are separated for scene and instance
        # Scene/canonical attends only to scene KV
        for i in range(1, num_instances):
            self_attn_instance = scaled_dot_product_attention(qp[:, i], kp[:, i], vp[:, i])
            output_list.append(self_attn_instance)

        # Stitch back to (B, L, H, C) → (B, L, C_all) → linear proj
        h = torch.stack(output_list, dim=1)  # (bp, num_instances, L, H, C)
        h = rearrange(h, 'bp ni L h c -> (bp ni) L h c')
        h = h.reshape(B, L, -1)
        h = self.to_out(h)
        return h

    def self_attn_join_external(self, x: torch.Tensor, external_tokens: Union[torch.Tensor, List[torch.Tensor]], indices: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Self-attention where queries come from x, and keys/values are augmented
        with one or more external token sequences. All projections (Q/K/V) use
        this module's own projection weights to keep them in the same space.

        Args:
            x: (B, Lq, C) queries from the current stream
            external_tokens: either a tensor (B, Lext, C) or a list of tensors
                              each of shape (B, Lext_i, C)
            indices: optional rotary indices
        Returns:
            (B, Lq, C) attended output
        """
        assert self._type == "self", "self_attn_join_external is only valid for self-attention"

        if isinstance(external_tokens, torch.Tensor):
            external_list: List[torch.Tensor] = [external_tokens]
        else:
            external_list = list(external_tokens)

        B, Lq, C = x.shape

        # Project Q/K/V for x
        qkv = self.to_qkv(x).reshape(B, Lq, 3, self.num_heads, -1)
        q, k, v = qkv.unbind(dim=2)

        if self.use_rope:
            q, k = self.rope(q, k, indices)

        # Optional Q/K RMSNorm
        if self.qk_rms_norm:
            q = self.q_rms_norm(q)
            k = self.k_rms_norm(k)

        # Project only K/V for external tokens using the SAME to_qkv weights
        k_ext_list: List[torch.Tensor] = []
        v_ext_list: List[torch.Tensor] = []
        for i, ext in enumerate(external_list):
            assert ext.dim() == 3, f"external token must be 3D (B, L, C), got {ext.shape}"
            assert ext.shape[0] == B, f"Batch size mismatch: ext B={ext.shape[0]} vs x B={B}"
            # Do not alter raw external token content; avoid adding source/type embedding to ext tokens
            ext_qkv = self.to_qkv(ext).reshape(ext.shape[0], ext.shape[1], 3, self.num_heads, -1)
            _, k_ext, v_ext = ext_qkv.unbind(dim=2)
            if self.use_rope:
                # apply RoPE to external K; use K as both inputs to get rotated K
                _, k_ext = self.rope(k_ext, k_ext, indices)
            if self.qk_rms_norm:
                k_ext = self.k_rms_norm(k_ext)

            if self.use_positional_encoding:
                # Optional per-head K type bias (vector) applied after RoPE/RMSNorm
                if getattr(self, 'enable_ext_k_bias', False):
                    bias_vec = torch.tanh(self.k_type_bias[i])[None, None, :, :].to(k_ext.dtype)
                    k_ext = k_ext + self.ext_k_bias_scale * bias_vec

                # Optional per-head gate to modulate influence of external keys only (values unchanged)
                if getattr(self, 'enable_ext_gate', False):
                    alpha = torch.sigmoid(self.ext_gate[i])[None, None, :, None].to(k_ext.dtype)
                    k_ext = k_ext * alpha

            k_ext_list.append(k_ext)
            v_ext_list.append(v_ext)

        # Concatenate K/V along sequence dimension
        if len(k_ext_list) > 0:
            k_cat = torch.cat([k] + k_ext_list, dim=1)
            v_cat = torch.cat([v] + v_ext_list, dim=1)
        else:
            k_cat, v_cat = k, v

        # Attention and output
        h = scaled_dot_product_attention(q, k_cat, v_cat)
        h = h.reshape(B, Lq, -1)
        h = self.to_out(h)
        return h