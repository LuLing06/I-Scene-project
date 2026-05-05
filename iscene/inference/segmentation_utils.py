from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np
from PIL import Image


def load_rgb_image(image_path: Union[str, Path]) -> Image.Image:
    """Load an RGB image and handle alpha / transparency consistently."""
    img = Image.open(image_path)
    if img.mode in ("RGBA", "LA") or ("transparency" in img.info):
        rgba = img.convert("RGBA")
        background = Image.new("RGBA", rgba.size, (0, 0, 0, 0))
        return Image.alpha_composite(background, rgba).convert("RGB")
    return img.convert("RGB")


def segmentation_to_id_map(segmentation: Image.Image) -> np.ndarray:
    """Decode an instance segmentation image into one integer label per pixel."""
    seg_array = np.array(segmentation)
    if seg_array.ndim == 2:
        return seg_array.astype(np.uint32)

    if seg_array.ndim == 3 and seg_array.shape[2] >= 1:
        channels = seg_array[..., :3].astype(np.uint32)
        if channels.shape[2] == 1:
            return channels[..., 0]

        r = channels[..., 0]
        g = channels[..., 1]
        b = channels[..., 2] if channels.shape[2] >= 3 else np.zeros_like(r)

        if np.array_equal(r, g) and np.array_equal(r, b):
            return r

        packed_rg = r + (g << 8)
        packed_rgb = packed_rg + (b << 16)
        rg_ids = np.unique(packed_rg)
        rgb_ids = np.unique(packed_rgb)

        # Preserve the legacy 16-bit R/G packed format when B carries no label
        # information. Use full RGB packing for color-coded masks so blue-only
        # labels are not dropped and distinct colors are not merged.
        if np.any(b != 0) or len(rgb_ids) != len(rg_ids):
            return packed_rgb
        return packed_rg

    return np.zeros(seg_array.shape[:2], dtype=np.uint32)


def load_scene_and_instance_masks(
    rgb_image_path: Union[str, Path],
    segmentation_path: Union[str, Path],
) -> tuple[Image.Image, list[Image.Image], list[int]]:
    """
    Load one scene RGB image and split a multi-label segmentation into per-instance masks.

    The segmentation can be single-channel label IDs, palette IDs, packed 16-bit
    R/G IDs, or RGB color-coded instance IDs.
    """
    segmentation = Image.open(segmentation_path)
    scene_rgb = load_rgb_image(rgb_image_path).resize(segmentation.size)

    id_map = segmentation_to_id_map(segmentation)

    label_ids = np.unique(id_map)
    label_ids = sorted(int(label_id) for label_id in label_ids[label_ids > 0].tolist())

    instance_masks: list[Image.Image] = []
    for label_id in label_ids:
        mask = np.zeros_like(id_map, dtype=np.uint8)
        mask[id_map == label_id] = 255
        instance_masks.append(Image.fromarray(mask))

    return scene_rgb, instance_masks, label_ids
