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


def load_scene_and_instance_masks(
    rgb_image_path: Union[str, Path],
    segmentation_path: Union[str, Path],
) -> tuple[Image.Image, list[Image.Image], list[int]]:
    """
    Load one scene RGB image and split a multi-label segmentation into per-instance masks.

    The segmentation can be either single-channel label IDs or packed 16-bit IDs
    encoded as R + (G << 8), which is the format used by the bundled example.
    """
    segmentation = Image.open(segmentation_path)
    scene_rgb = load_rgb_image(rgb_image_path).resize(segmentation.size)

    seg_array = np.array(segmentation)
    if seg_array.ndim == 2:
        id_map = seg_array.astype(np.uint32)
    elif seg_array.ndim == 3 and seg_array.shape[2] >= 2:
        id_map = seg_array[..., 0].astype(np.uint32) + (seg_array[..., 1].astype(np.uint32) << 8)
    else:
        id_map = np.zeros(seg_array.shape[:2], dtype=np.uint32)

    label_ids = np.unique(id_map)
    label_ids = sorted(int(label_id) for label_id in label_ids[label_ids > 0].tolist())

    instance_masks: list[Image.Image] = []
    for label_id in label_ids:
        mask = np.zeros_like(id_map, dtype=np.uint8)
        mask[id_map == label_id] = 255
        instance_masks.append(Image.fromarray(mask))

    return scene_rgb, instance_masks, label_ids
