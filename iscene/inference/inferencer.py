from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
from PIL import Image
from plyfile import PlyData, PlyElement
import torch
import trimesh
from tqdm import tqdm

from ..trellis.pipelines import TrellisImageTo3DSceneContextPipeline
from ..trellis.modules import sparse as sp

from .segmentation_utils import load_scene_and_instance_masks, segmentation_to_id_map


DEFAULT_BASE_MODEL_ID = "microsoft/TRELLIS-image-large"
SPARSE_STRUCTURE_SAMPLER_PARAMS = {"steps": 25, "cfg_strength": 3.0}
SLAT_SAMPLER_PARAMS = {"steps": 25, "cfg_strength": 3.0}


def _resolve_package_file(model_id_or_path: str | Path, filename: str, revision: str | None = None) -> Path:
    root = Path(model_id_or_path).expanduser()
    local_path = root / filename
    if local_path.exists():
        return local_path

    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(str(model_id_or_path), filename, revision=revision))


class ISceneInferencer:
    def __init__(
        self,
        model_id_or_path: str | Path,
        *,
        base_model_id: str | Path | None = None,
        revision: str | None = None,
        base_revision: str | None = None,
    ):
        self.model_id_or_path = str(model_id_or_path)
        self.base_model_id = str(base_model_id) if base_model_id is not None else None
        self.revision = revision
        self.base_revision = base_revision
        self.pipeline = None

    @classmethod
    def from_pretrained(
        cls,
        model_id_or_path: str | Path,
        *,
        base_model_id: str | Path | None = None,
        revision: str | None = None,
        base_revision: str | None = None,
    ) -> "ISceneInferencer":
        return cls(
            model_id_or_path,
            base_model_id=base_model_id,
            revision=revision,
            base_revision=base_revision,
        )

    def _load_release_metadata(self) -> dict:
        metadata_path = _resolve_package_file(self.model_id_or_path, "iscene_config.json", revision=self.revision)
        with open(metadata_path, "r") as f:
            metadata = json.load(f)

        required_keys = {
            "base_model_id",
            "config_file",
            "denoiser_checkpoint",
            "image_conditioner_checkpoint",
        }
        missing = sorted(required_keys - set(metadata))
        if missing:
            raise ValueError(f"IScene model package is missing required metadata keys: {missing}")
        return metadata

    def setup_pipeline(self):
        metadata = self._load_release_metadata()
        config_file = _resolve_package_file(self.model_id_or_path, metadata["config_file"], revision=self.revision)
        denoiser_checkpoint = _resolve_package_file(
            self.model_id_or_path,
            metadata["denoiser_checkpoint"],
            revision=self.revision,
        )
        image_conditioner_checkpoint = _resolve_package_file(
            self.model_id_or_path,
            metadata["image_conditioner_checkpoint"],
            revision=self.revision,
        )
        base_model_id = self.base_model_id or metadata.get("base_model_id", DEFAULT_BASE_MODEL_ID)

        pipeline, cfg = TrellisImageTo3DSceneContextPipeline.from_pretrained(
            str(base_model_id),
            config_file=config_file,
            denoiser_checkpoint=denoiser_checkpoint,
            image_conditioner_checkpoint=image_conditioner_checkpoint,
            revision=self.base_revision,
        )
        pipeline.cuda()
        pipeline.set_exp_cfg(cfg)
        return pipeline

    def infer_and_save_scene(
        self,
        scene_rgb_path: str | Path,
        instance_seg_path: str | Path,
        output_dir: str | Path,
        overwrite: bool = True,
        save_dbg: bool = False,
        simplify: float = 0.95,
        only_3dgs: bool = False,
        seed: int = 42,
        verbose: bool = False,
    ) -> None:
        scene_results = self.infer_scene_instances(
            scene_rgb_path,
            instance_seg_path,
            seed=seed,
            only_3dgs=only_3dgs,
            save_dbg=save_dbg,
            verbose=verbose,
        )
        self.save_scene_outputs(
            scene_results,
            output_dir,
            overwrite=overwrite,
            save_dbg=save_dbg,
            simplify=simplify,
            only_3dgs=only_3dgs,
            verbose=verbose,
        )

    @staticmethod
    def _prepare_instance_inputs(
        scene_rgb_path: str | Path,
        instance_seg_path: str | Path,
        input_loader=load_scene_and_instance_masks,
    ):
        scene_rgb, instance_masks, label_ids = input_loader(
            scene_rgb_path,
            instance_seg_path,
        )
        scene_mask = (segmentation_to_id_map(Image.open(instance_seg_path)) > 0).astype("uint8") * 255
        scene_mask_pil = Image.fromarray(scene_mask)
        return scene_rgb, instance_masks, scene_mask_pil, label_ids

    @staticmethod
    @torch.no_grad()
    def _sample_sparse_structure(
        pipeline,
        *,
        scene_rgb,
        scene_mask,
        instance_masks,
        seed: int,
        sparse_structure_sampler_params: dict,
        collect_debug: bool,
        verbose: bool,
    ) -> dict | None:
        if scene_rgb is None or not instance_masks:
            logging.warning("Empty input lists for sparse-structure inference.")
            return None

        preprocessed_list = []
        dbg_rets = [] if collect_debug else None
        for instance_mask in instance_masks:
            preprocessed, dbg_ret = pipeline.preprocess_image(
                scene_rgb,
                scene_mask,
                instance_mask,
                return_debug=collect_debug,
            )
            preprocessed_list.append(preprocessed)
            if collect_debug and dbg_rets is not None:
                dbg_rets.append(dbg_ret)

        exp_setting = getattr(pipeline.exp_cfg.dataset.args, "exp_setting", "")
        slot_names = ["scene_space_instance"]
        if "global" in exp_setting:
            slot_names.append("scene_space_scene")
        if "local" in exp_setting:
            slot_names.append("canonical_space_instance")

        ss_cond, slat_cond, resolved_batch_size, num_slots = pipeline.get_cond_batch(preprocessed_list)
        if len(slot_names) != num_slots:
            slot_names = [f"slot_{i}" for i in range(num_slots)]

        torch.manual_seed(seed)
        coords = pipeline.sample_sparse_structure(
            ss_cond,
            num_samples=resolved_batch_size,
            sampler_params=sparse_structure_sampler_params,
            verbose=verbose,
        )

        results = {
            "coords": coords,
            "num_instances": resolved_batch_size,
            "num_slots": num_slots,
            "slot_names": slot_names,
            "slat_cond": slat_cond,
        }
        if collect_debug:
            results["dbg_ret_list"] = dbg_rets
        return results

    @torch.no_grad()
    def infer_scene_instances(
        self,
        scene_rgb_path: str | Path,
        instance_seg_path: str | Path,
        seed: int = 42,
        only_3dgs: bool = False,
        save_dbg: bool = False,
        verbose: bool = False,
    ):
        scene_rgb, instance_masks, scene_mask_pil, label_ids = self._prepare_instance_inputs(
            scene_rgb_path,
            instance_seg_path,
        )

        if not instance_masks:
            logging.warning("No foreground instances found in segmentation.")
            return None

        if self.pipeline is None:
            self.pipeline = self.setup_pipeline()

        stage1_results = self._sample_sparse_structure(
            self.pipeline,
            scene_rgb=scene_rgb,
            scene_mask=scene_mask_pil,
            instance_masks=instance_masks,
            seed=seed,
            sparse_structure_sampler_params=SPARSE_STRUCTURE_SAMPLER_PARAMS,
            collect_debug=save_dbg,
            verbose=verbose,
        )
        if stage1_results is None:
            return None

        coords = stage1_results["coords"]
        slat = self.pipeline.sample_slat(
            stage1_results["slat_cond"],
            coords,
            sampler_params=SLAT_SAMPLER_PARAMS,
            verbose=verbose,
        )

        num_instances = stage1_results["num_instances"]
        num_slots = stage1_results["num_slots"]
        slot_names = stage1_results["slot_names"]
        total_slots = num_instances * num_slots
        scene_slot_idx = slot_names.index("scene_space_scene") if "scene_space_scene" in slot_names else -1
        skipped_slot_ids = {
            instance_idx * num_slots + scene_slot_idx
            for instance_idx in range(num_instances)
        } if scene_slot_idx >= 0 else set()

        unique_batch_ids = torch.unique(slat.coords[:, 0]).sort()[0]
        decode_formats = ["gaussian"] if only_3dgs else ["mesh", "gaussian"]
        decoded_results = {fmt: [None] * total_slots for fmt in decode_formats}
        for bid in tqdm(unique_batch_ids, desc="Decoding assets", disable=not verbose):
            bid_int = int(bid.item())
            if bid_int in skipped_slot_ids:
                continue
            mask = slat.coords[:, 0] == bid
            sample_coords = slat.coords[mask].clone()
            sample_coords[:, 0] = 0
            sample_slat = sp.SparseTensor(
                feats=slat.feats[mask],
                coords=sample_coords,
            )
            sample_decoded = self.pipeline.decode_slat(sample_slat, decode_formats)
            for fmt, values in decoded_results.items():
                if fmt in sample_decoded:
                    values[bid_int] = sample_decoded[fmt]

        scene_results = {
            **decoded_results,
            "coords": coords,
            "num_instances": num_instances,
            "num_slots": num_slots,
            "slot_names": slot_names,
        }
        if save_dbg:
            scene_results["dbg_ret_list"] = stage1_results.get("dbg_ret_list", [])

        scene_results["label_ids"] = label_ids
        if save_dbg:
            scene_results["scene_rgb"] = scene_rgb
            scene_results["instance_masks"] = instance_masks
        return scene_results

    def save_scene_outputs(
        self,
        scene_results,
        output_dir: str | Path,
        overwrite: bool = True,
        save_dbg: bool = False,
        simplify: float = 0.95,
        only_3dgs: bool = False,
        verbose: bool = False,
    ) -> None:
        if scene_results is None:
            return

        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        if overwrite:
            for stale_scene_slot in out_dir.glob("instance_*_scene_space_scene.*"):
                stale_scene_slot.unlink()
            for stale_instance_slot in out_dir.glob("instance_*_scene_space_instance.*"):
                stale_instance_slot.unlink()
            for stale_scene_slot in out_dir.glob("scene_space_scene.*"):
                stale_scene_slot.unlink()

        label_ids = scene_results.get("label_ids", [])
        slot_names = scene_results.get("slot_names", [])
        num_instances = int(scene_results.get("num_instances", len(label_ids)))
        num_slots = int(scene_results.get("num_slots", len(slot_names) if slot_names else 0))
        meshes = scene_results.get("mesh")
        gaussians = scene_results.get("gaussian")
        coords = scene_results.get("coords")

        if gaussians is None:
            raise ValueError("scene_results must contain gaussian outputs.")
        if not only_3dgs and meshes is None:
            raise ValueError("scene_results must contain mesh outputs when only_3dgs=False.")

        if num_slots <= 0:
            num_slots = max(1, len(gaussians) // max(num_instances, 1))
        if not slot_names or len(slot_names) != num_slots:
            slot_names = [f"slot_{i}" for i in range(num_slots)]

        scene_slot_idx = slot_names.index("scene_space_scene") if "scene_space_scene" in slot_names else -1
        instance_slot_idx = slot_names.index("scene_space_instance") if "scene_space_instance" in slot_names else 0

        if only_3dgs:
            instance_ply_paths: list[str] = []
            for instance_idx in tqdm(range(num_instances), desc="Saving Gaussian assets", disable=not verbose):
                label_id = label_ids[instance_idx] if instance_idx < len(label_ids) else instance_idx
                for slot_idx in range(num_slots):
                    if slot_idx != instance_slot_idx or slot_idx == scene_slot_idx:
                        continue

                    flat_idx = instance_idx * num_slots + slot_idx
                    ply_path = out_dir / f"instance_{int(label_id):02d}.ply"
                    if ply_path.exists() and not overwrite:
                        instance_ply_paths.append(str(ply_path))
                        continue

                    gaussian = gaussians[flat_idx]
                    if gaussian is None:
                        continue

                    gaussian[0].save_ply(str(ply_path))
                    instance_ply_paths.append(str(ply_path))

            if instance_ply_paths:
                scene_ply_path = out_dir / "scene_pred.ply"
                if overwrite or not scene_ply_path.exists():
                    merge_gaussian_ply_files(instance_ply_paths, str(scene_ply_path))
        else:
            from ..trellis.utils import postprocessing_utils

            instance_glbs: list[Path] = []

            for instance_idx in tqdm(range(num_instances), desc="Exporting GLB assets", disable=not verbose):
                label_id = label_ids[instance_idx] if instance_idx < len(label_ids) else instance_idx
                for slot_idx in range(num_slots):
                    if slot_idx != instance_slot_idx or slot_idx == scene_slot_idx:
                        continue

                    flat_idx = instance_idx * num_slots + slot_idx
                    out_path = out_dir / f"instance_{int(label_id):02d}.glb"
                    if out_path.exists() and not overwrite:
                        instance_glbs.append(out_path)
                        continue

                    gaussian = gaussians[flat_idx]
                    mesh = meshes[flat_idx]
                    if gaussian is None or mesh is None:
                        continue

                    glb = postprocessing_utils.to_glb(
                        gaussian[0],
                        mesh[0],
                        simplify=simplify,
                        texture_size=1024,
                        verbose=False,
                    )
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    glb.export(str(out_path))
                    instance_glbs.append(out_path)

            if instance_glbs:
                scene_mesh = self._merge_instance_glbs_to_scene(sorted(instance_glbs))
                scene_mesh.export(str(out_dir / "scene_pred.glb"))

        if save_dbg:
            self._save_debug_outputs(
                scene_results,
                out_dir,
                label_ids=label_ids,
                slot_names=slot_names,
                num_slots=num_slots,
                coords=coords,
            )

    def _save_debug_outputs(
        self,
        scene_results,
        out_dir: Path,
        *,
        label_ids: list[int],
        slot_names: list[str],
        num_slots: int,
        coords,
    ) -> None:
        scene_rgb = scene_results.get("scene_rgb")
        instance_masks = scene_results.get("instance_masks")
        dbg_ret_list = scene_results.get("dbg_ret_list", [])
        num_instances = int(scene_results.get("num_instances", len(label_ids)))

        for instance_idx in range(num_instances):
            label_id = label_ids[instance_idx] if instance_idx < len(label_ids) else instance_idx

            if scene_rgb is not None:
                scene_rgb.save(str(out_dir / f"instance_{int(label_id):02d}_scene_rgb.png"))
            if instance_masks is not None and instance_idx < len(instance_masks):
                instance_masks[instance_idx].save(str(out_dir / f"instance_{int(label_id):02d}_instance_mask.png"))

            if dbg_ret_list and instance_idx < len(dbg_ret_list):
                dbg_ret = dbg_ret_list[instance_idx]
                if "instance_rgb_canonical_tensor" in dbg_ret:
                    canonical_np = dbg_ret["instance_rgb_canonical_tensor"].cpu().numpy().transpose(1, 2, 0)
                    canonical_np = np.clip(canonical_np * 255.0, 0, 255).astype(np.uint8)
                    Image.fromarray(canonical_np).save(
                        str(out_dir / f"instance_{int(label_id):02d}_canonical_space_instance_rgb.png")
                    )

            if coords is not None:
                for slot_idx in range(num_slots):
                    flat_idx = instance_idx * num_slots + slot_idx
                    coord_path = out_dir / f"instance_{int(label_id):02d}_{slot_names[slot_idx]}_coords.ply"
                    save_sparse_coords_as_ply(coords[coords[:, 0] == flat_idx], str(coord_path))

    @staticmethod
    def _merge_instance_glbs_to_scene(instance_mesh_paths):
        aggregated = trimesh.Scene()

        for idx, mesh_path in enumerate(sorted(Path(p) for p in instance_mesh_paths)):
            try:
                loaded = trimesh.load(str(mesh_path))
            except Exception as exc:
                logging.warning("Failed to load %s for scene aggregation: %s", mesh_path, exc)
                continue

            stem = mesh_path.stem
            if hasattr(loaded, "geometry"):
                for sub_idx, (sub_name, geometry) in enumerate(loaded.geometry.items()):
                    base_name = f"{stem}_{sub_name}" if sub_name else stem
                    node_name = base_name if base_name not in aggregated.geometry else f"{base_name}_{sub_idx}"
                    aggregated.add_geometry(geometry, node_name=node_name)
            else:
                node_name = stem if stem not in aggregated.geometry else f"{stem}_{idx}"
                aggregated.add_geometry(loaded, node_name=node_name)

        return aggregated


def save_sparse_coords_as_ply(coords, output_path: str, resolution: int = 64) -> None:
    spatial_coords = coords[:, 1:].float().cpu().numpy()
    points = (spatial_coords + 0.5) / resolution * 2.0 - 1.0
    points = points[:, [0, 2, 1]]
    points[:, 2] = -points[:, 2]
    trimesh.points.PointCloud(points).export(output_path)


def merge_gaussian_ply_files(ply_paths: list[str], output_path: str) -> None:
    all_vertices = []
    for ply_path in ply_paths:
        if not Path(ply_path).exists():
            continue
        try:
            plydata = PlyData.read(str(ply_path))
        except Exception as exc:
            logging.warning("Failed to read %s: %s", ply_path, exc)
            continue
        all_vertices.append(plydata["vertex"].data)

    if not all_vertices:
        return

    merged = np.concatenate(all_vertices)
    PlyData([PlyElement.describe(merged, "vertex")]).write(output_path)
