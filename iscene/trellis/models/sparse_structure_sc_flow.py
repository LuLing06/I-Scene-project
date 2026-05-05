from typing import *
import torch
import torch.nn.functional as F
from ..modules.utils import convert_module_to_f16
from ..modules.spatial import patchify, unpatchify
from pathlib import Path
from .sparse_structure_flow import SparseStructureFlowModel

class SparseStructureSceneContextFlowModel(SparseStructureFlowModel):
    def __init__(
        self,
        resolution: int,
        in_channels: int,
        model_channels: int,
        cond_channels: int,
        out_channels: int,
        num_blocks: int,
        num_heads: Optional[int] = None,
        num_head_channels: Optional[int] = 64,
        mlp_ratio: float = 4,
        patch_size: int = 2,
        pe_mode: Literal["ape", "rope"] = "ape",
        use_fp16: bool = False,
        use_checkpoint: bool = False,
        share_mod: bool = False,
        qk_rms_norm: bool = False,
        qk_rms_norm_cross: bool = False,
        pretrained_base: Optional[str] = None,
        scene_context_attn_num: int = 5,
        learning_pattern: Literal['full-finetune'] = 'full-finetune',
        exp_setting: str = "global local",
        type_embedding_type = None,
        k_bias_scale = 0.2,
    ):
        super().__init__(resolution, in_channels, model_channels, cond_channels, out_channels, num_blocks, num_heads, num_head_channels, mlp_ratio, patch_size, pe_mode, use_fp16, use_checkpoint, share_mod, qk_rms_norm, qk_rms_norm_cross)

        assert pretrained_base is not None, 'pretrained_base is required for SparseStructureSceneContextFlowModel'
        assert Path(pretrained_base).exists(), f'Pretrained base model {pretrained_base} not found'
        self.scene_context_attn_num = scene_context_attn_num

        # load the base model
        if Path(pretrained_base).suffix == '.pt':
            self.load_state_dict(torch.load(pretrained_base, map_location='cpu'), strict=True)
        elif Path(pretrained_base).suffix == '.safetensors':
            from safetensors.torch import load_file
            self.load_state_dict(load_file(pretrained_base), strict=True)
        else:
            raise ValueError(f'Invalid pretrained base model {pretrained_base}')

        # hijack some blocks to use scene context attention
        block_num = len(self.blocks)
        start_idx = block_num // 2 - scene_context_attn_num // 2
        for i in range(scene_context_attn_num):
            self.blocks[start_idx + i].is_scene_context = True
            self.blocks[start_idx + i].num_instances = len(exp_setting.split(' ')) + 1
            if type_embedding_type is not None:
                enable_gate = 'enable_gate' in type_embedding_type
                enable_k_bias = 'enable_k_bias' in type_embedding_type
                k_bias_scale = k_bias_scale
                self.blocks[start_idx + i].self_attn.initialize_positional_encoding(self.blocks[start_idx + i].num_instances - 1,
                                                                                    enable_gate=enable_gate,
                                                                                    enable_k_bias=enable_k_bias,
                                                                                    k_bias_scale=k_bias_scale)

        if use_fp16:
            self.convert_to_fp16()

        if learning_pattern != 'full-finetune':
            raise ValueError(f'Unsupported learning pattern for release inference: {learning_pattern}')


    def convert_to_fp16(self) -> None:
        """
        Convert the torso of the model to float16.
        """
        for block in self.blocks:
            block.apply(convert_module_to_f16)
    def forward(self, x: torch.Tensor, t: torch.Tensor, cond: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """
        x: B, N, C, [resolution, resolution, resolution]
        cond: B, N, C, H, W
        """
        B, N, C, *rest = x.shape
        x = x.view(B * N, C, *rest)

        B, N, T, C = cond.shape
        cond = cond.view(B * N, T, C)

        t = t.repeat_interleave(N, dim=0)
        h = patchify(x, self.patch_size)
        h = h.view(*h.shape[:2], -1).permute(0, 2, 1).contiguous()

        h = self.input_layer(h)
        h = h + self.pos_emb[None]
        t_emb = self.t_embedder(t)
        if self.share_mod:
            t_emb = self.adaLN_modulation(t_emb)
        t_emb = t_emb.type(self.dtype)
        h = h.type(self.dtype)
        cond = cond.type(self.dtype)

        for block in self.blocks:
            h = block(x=h, mod=t_emb, context=cond)

        h = h.type(x.dtype)
        h = F.layer_norm(h, h.shape[-1:])
        h = self.out_layer(h)
        h = h.permute(0, 2, 1).view(h.shape[0], h.shape[2], *[self.resolution // self.patch_size] * 3)
        h = unpatchify(h, self.patch_size).contiguous()
        h = h.view(B, N, *h.shape[1:])
        return h
