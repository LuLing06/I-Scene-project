from typing import *
import json
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from .base import Pipeline
from . import samplers
from ..modules import sparse as sp
import cv2
from pathlib import Path
from omegaconf import OmegaConf
from .. import models


def _resolve_model_file(model_id_or_path: str, filename: str, revision: str | None = None) -> Path:
    root = Path(model_id_or_path).expanduser()
    local_path = root / filename
    if local_path.exists():
        return local_path

    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(model_id_or_path, filename, revision=revision))

class TrellisImageTo3DSceneContextPipeline(Pipeline):
    """
    Pipeline for inferring Trellis image-to-3D models.

    Args:
        models (dict[str, nn.Module]): The models to use in the pipeline.
        sparse_structure_sampler (samplers.Sampler): The sampler for the sparse structure.
        slat_sampler (samplers.Sampler): The sampler for the structured latent.
        slat_normalization (dict): The normalization parameters for the structured latent.
        image_cond_model (str): The name of the image conditioning model.
    """
    def __init__(
        self,
        models: dict[str, nn.Module] = None,
        sparse_structure_sampler: samplers.Sampler = None,
        slat_sampler: samplers.Sampler = None,
        slat_normalization: dict = None,
    ):
        if models is None:
            return
        super().__init__(models)
        self.sparse_structure_sampler = sparse_structure_sampler
        self.slat_sampler = slat_sampler
        self.sparse_structure_sampler_params = {}
        self.slat_sampler_params = {}
        self.slat_normalization = slat_normalization

    @staticmethod
    def from_pretrained(
        path: str,
        *,
        config_file: str | Path,
        denoiser_checkpoint: str | Path,
        image_conditioner_checkpoint: str | Path,
        revision: str | None = None,
    ) -> "TrellisImageTo3DSceneContextPipeline":
        """
        Load the TRELLIS base models plus IScene release weights.
        """
        pipeline_config_file = _resolve_model_file(path, "pipeline.json", revision=revision)
        with open(pipeline_config_file, "r") as f:
            args = json.load(f)["args"]

        required_model_keys = (
            "sparse_structure_decoder",
            "slat_decoder_gs",
            "slat_decoder_mesh",
            "slat_flow_model",
        )
        loaded_models = {}
        for model_key in required_model_keys:
            model_name = args["models"][model_key]
            loaded_models[model_key] = models.from_pretrained(f"{path}/{model_name}", revision=revision)

        new_pipeline = TrellisImageTo3DSceneContextPipeline(loaded_models)
        new_pipeline._pretrained_args = args

        new_pipeline.sparse_structure_sampler = getattr(samplers, args['sparse_structure_sampler']['name'])(**args['sparse_structure_sampler']['args'])
        new_pipeline.sparse_structure_sampler_params = args['sparse_structure_sampler']['params']

        new_pipeline.slat_sampler = getattr(samplers, args['slat_sampler']['name'])(**args['slat_sampler']['args'])
        new_pipeline.slat_sampler_params = args['slat_sampler']['params']

        new_pipeline.slat_normalization = args['slat_normalization']

        cfg = OmegaConf.load(config_file)
        base_sparse_flow = _resolve_model_file(
            path,
            f"{args['models']['sparse_structure_flow_model']}.safetensors",
            revision=revision,
        )

        denoiser_args = OmegaConf.to_container(cfg.models.denoiser.args, resolve=True)
        denoiser_args["pretrained_base"] = str(base_sparse_flow)
        img_conditioner_args = OmegaConf.to_container(cfg.models.img_conditioner.args, resolve=True)

        sparse_structure_flow_model = getattr(models, cfg.models.denoiser.name)(**denoiser_args)
        img_conditioner = getattr(models, cfg.models.img_conditioner.name)(**img_conditioner_args)

        sparse_structure_flow_model.load_state_dict(torch.load(denoiser_checkpoint, map_location='cpu'))
        img_conditioner.load_state_dict(torch.load(image_conditioner_checkpoint, map_location='cpu'))
        new_pipeline.models['sparse_structure_flow_model'] = sparse_structure_flow_model
        new_pipeline.models['img_conditioner'] = img_conditioner

        return new_pipeline, cfg

    def set_exp_cfg(self, cfg):
        self.exp_cfg = cfg

    def preprocess_image(
        self,
        scene_rgb: Image.Image,
        scene_mask: Image.Image,
        instance_mask: Image.Image,
        return_debug: bool = False,
    ) -> Image.Image:
        """
        Preprocess the input image.
        """
        def crop_bbox(alpha_np):
            bbox = np.argwhere(alpha_np > 0.8)
            bbox = np.min(bbox[:, 1]), np.min(bbox[:, 0]), np.max(bbox[:, 1]), np.max(bbox[:, 0])
            center = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
            size = max(bbox[2] - bbox[0], bbox[3] - bbox[1])
            size = int(size * 1.2)
            bbox = center[0] - size // 2, center[1] - size // 2, center[0] + size // 2, center[1] + size // 2
            return bbox

        # binarize the scene_mask
        scene_mask = scene_mask.convert('L')
        scene_mask_np = np.array(scene_mask).astype(np.float32) / 255.0
        scene_mask_np[scene_mask_np >= 1/255.0] = 1
        scene_mask_np[scene_mask_np < 1/255.0] = 0
        bbox = crop_bbox(scene_mask_np)

        scene_rgb_cropped = scene_rgb.crop(bbox)
        scene_mask_cropped = scene_mask.crop(bbox)
        instance_mask_cropped = instance_mask.crop(bbox)

        # scene space scene rgb background removed
        scene_rgb_cropped_np = np.array(scene_rgb_cropped).astype(np.float32) / 255.0
        scene_mask_cropped_np = np.array(scene_mask_cropped).astype(np.float32) / 255.0
        scene_mask_cropped_np[scene_mask_cropped_np > 0.01] = 1
        scene_mask_cropped_np[scene_mask_cropped_np <= 0.01] = 0
        scene_rgb_cropped_np = (scene_rgb_cropped_np * scene_mask_cropped_np[..., None])[..., :3]

        # scene space instance rgb
        instance_mask_cropped_np = np.array(instance_mask_cropped.convert('L')).astype(np.float32) / 255.0
        instance_mask_cropped_np[instance_mask_cropped_np > 0.01] = 1
        instance_mask_cropped_np[instance_mask_cropped_np <= 0.01] = 0
        instance_rgb_cropped_np = (scene_rgb_cropped_np * instance_mask_cropped_np[..., None])[..., :3]

        # canonical space instance rgb needs to be cropped
        canonical_bbox = crop_bbox(instance_mask_cropped_np)
        instance_rgb_canonical_np = np.array(scene_rgb_cropped.crop(canonical_bbox).convert('RGB')).astype(np.float32) / 255.0
        instance_rgb_mask_canonical_np = np.array(instance_mask_cropped.crop(canonical_bbox).convert('L')).astype(np.float32) / 255.0
        instance_rgb_mask_canonical_np[instance_rgb_mask_canonical_np > 0.01] = 1
        instance_rgb_mask_canonical_np[instance_rgb_mask_canonical_np <= 0.01] = 0
        instance_rgb_canonical_np = instance_rgb_canonical_np * instance_rgb_mask_canonical_np[..., None]

        # prepare the output tensors
        # output is: scene_rgb, scene space instance_rgb, canonical space instance_rgb
        scene_rgb_cropped_np = np.clip(cv2.resize(scene_rgb_cropped_np, (518, 518), interpolation=cv2.INTER_LANCZOS4), 0, 1)
        instance_rgb_cropped_np = np.clip(cv2.resize(instance_rgb_cropped_np, (518, 518), interpolation=cv2.INTER_LANCZOS4), 0, 1)
        instance_rgb_canonical_np = np.clip(cv2.resize(instance_rgb_canonical_np, (518, 518), interpolation=cv2.INTER_LANCZOS4), 0, 1)

        scene_rgb_cropped_tensor = torch.from_numpy(scene_rgb_cropped_np).permute(2, 0, 1)
        instance_rgb_cropped_tensor = torch.from_numpy(instance_rgb_cropped_np).permute(2, 0, 1)
        instance_rgb_canonical_tensor = torch.from_numpy(instance_rgb_canonical_np).permute(2, 0, 1)

        # the order matters: instance_rgb_cropped_tensor, scene_rgb_cropped_tensor, instance_rgb_canonical_tensor
        cond_tensors = [instance_rgb_cropped_tensor]
        if 'global' in self.exp_cfg.dataset.args.exp_setting:
            cond_tensors.append(scene_rgb_cropped_tensor)

        if 'local' in self.exp_cfg.dataset.args.exp_setting:
            cond_tensors.append(instance_rgb_canonical_tensor)

        ret_tensor = torch.stack(cond_tensors, dim=0)

        if not return_debug:
            return ret_tensor, None

        dbg_ret = {
            'scene_rgb_cropped_tensor': scene_rgb_cropped_tensor,
            'instance_rgb_cropped_tensor': instance_rgb_cropped_tensor,
            'instance_rgb_canonical_tensor': instance_rgb_canonical_tensor,
        }
        return ret_tensor, dbg_ret


    @torch.no_grad()
    def encode_image(self, conditions: Union[torch.Tensor, list[Image.Image]]) -> torch.Tensor:
        """
        Encode the image.

        Args:
            image (Union[torch.Tensor, list[Image.Image]]): The image to encode

        Returns:
            torch.Tensor: The encoded features.
        """
        return self.models['img_conditioner'](conditions)

    def get_cond(self, conditions: Union[torch.Tensor]) -> dict:
        """
        Get the conditioning information for the model.

        Args:
            image (Union[torch.Tensor, list[Image.Image]]): The image prompts.

        Returns:
            dict: The conditioning information
        """
        # Note, condition has shape: B, N, 3, 518, 518
        # We view it as N x B, 3, 518, 518
        conditions = conditions.unsqueeze(0)
        B, N, C, H, W = conditions.shape
        conditions = conditions.view(B * N, C, H, W)
        cond = self.encode_image(conditions)
        B, T, C = cond.shape
        cond = cond.view(B//N, N, T, C)
        neg_cond = torch.zeros_like(cond)

        ss_cond = {
            'cond': cond,
            'neg_cond': neg_cond
        }
        slat_cond = {
            'cond': cond.view(B, T, C),
            'neg_cond': neg_cond.view(B, T, C)
        }
        return ss_cond, slat_cond

    def sample_sparse_structure(
        self,
        cond: dict,
        num_samples: int = 1,
        sampler_params: dict = {},
    ) -> torch.Tensor:
        """
        Sample sparse structures with the given conditioning.

        Args:
            cond (dict): The conditioning information.
            num_samples (int): The number of samples to generate.
            sampler_params (dict): Additional parameters for the sampler.
        """
        # Sample occupancy latent
        flow_model = self.models['sparse_structure_flow_model']
        reso = flow_model.resolution

        B, N, T, C = cond['cond'].shape
        num_instances = N
        single_noise = torch.randn(1, num_instances, flow_model.in_channels, reso, reso, reso).to(self.device)
        # copy the single_noise to each sample
        noise = single_noise.repeat(num_samples, 1, 1, 1, 1, 1)

        sampler_params = {**self.sparse_structure_sampler_params, **sampler_params}
        z_s = self.sparse_structure_sampler.sample(
            flow_model,
            noise,
            **cond,
            **sampler_params,
            verbose=False
        ).samples

        # z_s is a tensor of instance-scene sparse structures
        B, N, *rest = z_s.shape
        z_s = z_s.view(B * N, *rest)

        # Decode occupancy latent
        decoder = self.models['sparse_structure_decoder']
        coords = torch.argwhere(decoder(z_s)>0)[:, [0, 2, 3, 4]].int()

        return coords

    def decode_slat(
        self,
        slat: sp.SparseTensor,
        formats: List[str] = ['mesh', 'gaussian'],
    ) -> dict:
        """
        Decode the structured latent.

        Args:
            slat (sp.SparseTensor): The structured latent.
            formats (List[str]): The formats to decode the structured latent to.

        Returns:
            dict: The decoded structured latent.
        """
        ret = {}
        if 'mesh' in formats:
            ret['mesh'] = self.models['slat_decoder_mesh'](slat)
        if 'gaussian' in formats:
            ret['gaussian'] = self.models['slat_decoder_gs'](slat)
        return ret

    def sample_slat(
        self,
        cond: dict,
        coords: torch.Tensor,
        sampler_params: dict = {},
    ) -> sp.SparseTensor:
        """
        Sample structured latent with the given conditioning.

        Args:
            cond (dict): The conditioning information.
            coords (torch.Tensor): The coordinates of the sparse structure.
            sampler_params (dict): Additional parameters for the sampler.
        """
        # Sample structured latent
        flow_model = self.models['slat_flow_model']
        noise = sp.SparseTensor(
            feats=torch.randn(coords.shape[0], flow_model.in_channels).to(self.device),
            coords=coords,
        )
        sampler_params = {**self.slat_sampler_params, **sampler_params}

        slat = self.slat_sampler.sample(
            flow_model,
            noise,
            **cond,
            **sampler_params,
            verbose=False
        ).samples

        std = torch.tensor(self.slat_normalization['std'])[None].to(slat.device)
        mean = torch.tensor(self.slat_normalization['mean'])[None].to(slat.device)
        slat = slat * std + mean

        return slat

    def get_cond_batch(self, preprocessed_list: List[Union[torch.Tensor, dict]]) -> Tuple[dict, dict]:
        """
        Batched version of get_cond that handles B>1 samples.

        Args:
            preprocessed_list: List of preprocessed tensors/dicts from preprocess_image, one per instance.

        Returns:
            ss_cond: dict with 'cond' shape [B, Nslots, T, C] for sparse structure sampling
            slat_cond: dict with 'cond' shape [B*Nslots, T, C] for SLAT sampling
        """
        # Non-two-branch: preprocessed_list contains tensors [Nslots, 3, H, W]
        cond_batch = torch.stack(preprocessed_list, dim=0).to(self.device)  # [B, Nslots, 3, H, W]
        B, Nslots, Cc, H, W = cond_batch.shape
        cond_imgs_flat = cond_batch.view(B * Nslots, Cc, H, W)
        emb = self.encode_image(cond_imgs_flat)  # [B*Nslots, T, C]
        _, T, Ctok = emb.shape

        ss_pos = emb.view(B, Nslots, T, Ctok)
        ss_neg = torch.zeros_like(ss_pos)
        ss_cond = {'cond': ss_pos, 'neg_cond': ss_neg}

        slat_pos = emb.view(B * Nslots, T, Ctok)
        slat_neg = torch.zeros_like(slat_pos)
        slat_cond = {'cond': slat_pos, 'neg_cond': slat_neg}

        return ss_cond, slat_cond, B, Nslots
