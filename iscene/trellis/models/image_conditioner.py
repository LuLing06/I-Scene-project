import torch
import torch.nn as nn
from torchvision import transforms
import torch.nn.functional as F
import logging

from ..modules.utils import convert_module_to_f32
from ..utils import dist_utils

class ImageConditioner(nn.Module):
    def __init__(self, image_cond_model: str = 'dinov2_vitl14_reg', cond_in_channels: int = 10, use_fp16: bool = True):
        super().__init__()

        self.image_cond_model_name = image_cond_model
        self.cond_in_channels = cond_in_channels
        self._init_image_cond_model()

        if use_fp16:
            self.convert_to_fp16()
        self.dtype = torch.float16 if use_fp16 else torch.float32


    def convert_to_fp16(self):
        logging.info('Image conditioner does not support fp16, skip this.')


    def convert_to_fp32(self):
        logging.info('Image conditioner does not support fp32, skip this.')
        self.base_img_conditioner.apply(convert_module_to_f32)


    def forward(self, image: torch.Tensor):
        if isinstance(image, torch.Tensor):
            assert image.ndim == 4, "Image tensor should be batched (B, C, H, W)"
        elif isinstance(image, list):
            raise ValueError(f"Unsupported type of image: {type(image)}")
        else:
            raise ValueError(f"Unsupported type of image: {type(image)}")

        image = image.to(self.dtype).cuda()

        if image.shape[1] == 3:
            base_img = self.base_transform(image)
        else:
            # Handle multi-channel input (e.g. 7 channels: RGB + RGB + Mask)
            # We normalize every 3-channel block using ImageNet stats, and leave the rest as is.
            mean = torch.tensor([0.485, 0.456, 0.406], device=image.device, dtype=image.dtype).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225], device=image.device, dtype=image.dtype).view(1, 3, 1, 1)

            chunks = []
            for i in range(0, image.shape[1], 3):
                chunk = image[:, i:min(i+3, image.shape[1])]
                if chunk.shape[1] == 3:
                    chunk = (chunk - mean) / std
                chunks.append(chunk)
            base_img = torch.cat(chunks, dim=1)

        B, C, H, W = base_img.shape
        patchtokens = []

        features = self.base_img_conditioner(base_img, is_training=True)['x_prenorm']
        patchtokens = F.layer_norm(features, features.shape[-1:])
        return patchtokens


    def _init_image_cond_model(self):
        """
        Initialize the image conditioning model.
        """
        with dist_utils.local_master_first():
            dinov2_model = torch.hub.load('facebookresearch/dinov2', self.image_cond_model_name, pretrained=True)
        dinov2_model.eval().cuda()
        transform = transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.base_img_conditioner = dinov2_model
        self.base_transform = transform

        if self.cond_in_channels > 3:
            self.base_img_conditioner = self.expand_dinov2_model(self.base_img_conditioner, self.cond_in_channels)

        self.set_param_requires_grad(self.base_img_conditioner, False)


    def set_param_requires_grad(self, model, requires_grad: bool):
        for param in model.parameters():
            param.requires_grad = requires_grad


    def expand_dinov2_model(self, dinov2_model, cond_in_channels: int):
        """
        Expand the DINOv2 patch embedding to accept additional input channels.
        """

        # locate the patch-embedding projection conv for both hf Dinov2Model and torch.hub model
        if hasattr(dinov2_model, 'embeddings'):
            proj = dinov2_model.embeddings.patch_embeddings.projection
        elif hasattr(dinov2_model, 'patch_embed'):
            proj = dinov2_model.patch_embed.proj
        else:
            raise RuntimeError('Cannot locate patch-embedding projection in DINOv2 model.')

        if proj.weight.shape[1] < cond_in_channels:
            weight = proj.weight  # (out_channels, 3, k, k)

            extra = []
            channels_left = cond_in_channels - 3
            while channels_left > 0:
                take = min(3, channels_left)
                extra.append(weight[:, :take].clone())
                channels_left -= take

            new_weight = torch.cat([weight] + extra, dim=1)

            new_proj = torch.nn.Conv2d(
                in_channels=cond_in_channels,
                out_channels=weight.shape[0],
                kernel_size=proj.kernel_size,
                stride=proj.stride,
                padding=proj.padding,
                bias=(proj.bias is not None),
            )
            new_proj.weight.data = new_weight
            if proj.bias is not None:
                new_proj.bias.data = proj.bias.data.clone()

            # replace inside the model
            if hasattr(dinov2_model, 'embeddings'):
                dinov2_model.embeddings.patch_embeddings.projection = new_proj
            else:
                dinov2_model.patch_embed.proj = new_proj

        return dinov2_model
