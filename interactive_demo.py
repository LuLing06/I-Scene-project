"""Interactive I-Scene demo.

Run from the repository root:

    python interactive_demo.py
"""

from __future__ import annotations

import argparse
import os
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

import gradio as gr
import numpy as np
import torch
from gradio_image_prompter import ImagePrompter
from gradio_litmodel3d import LitModel3D
from PIL import Image
from transformers import AutoModelForMaskGeneration, AutoProcessor

from iscene.inference.inferencer import ISceneInferencer


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = "LuLing/IScene"
MODEL_ID = DEFAULT_MODEL
BASE_MODEL_ID: str | None = None
DEFAULT_SEED = 43
DEFAULT_SIMPLIFY = 0.95
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "outputs" / "demo"
UPLOAD_ROOT = DEFAULT_OUTPUT_ROOT / "_uploads"
TARGET_SIZE = (512, 512)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32

SAM_MODELS = {
    "sam-vit-huge (best quality, 636M)": "facebook/sam-vit-huge",
    "sam-vit-large (balanced, 308M)": "facebook/sam-vit-large",
    "sam-vit-base (fastest, 91M)": "facebook/sam-vit-base",
}

MARKDOWN = """
# I-Scene Interactive Demo

Generate a 3D scene from one image.

Workflow:
1. Pick an example, or upload an image and draw boxes around objects.
2. Use the example mask, or click **Run SAM Segmentation** to create a mask.
3. Click **Generate Gaussian Splatting Preview** to create and preview `scene_pred.ply`.
4. Click **Generate GLB** only when you need mesh assets.
5. To save each instance in the scene, run the inference code with the same RGB/mask; `run_inference.py` writes per-instance assets alongside the scene output.

Note: The first run may be slow because the model checkpoint needs to be downloaded and cached.
"""

EXAMPLE_ORDER = [
    "Scenethesis/SAM-3D-testing-case_rgb.png",
    "Gen3DSR/Gen3DSR_scene1_rgb.png",
    "MIDI-example/cartoon_style_07_rgb.png",
    "Scenethesis/children_playroom2_rgb.png",
    "Scenethesis/scenethesis-reading-corner-rgb.png",
    "DL3DV/DL3DV-garden-rgb.png",
    "DL3DV/DL3DV-table-chair-set-rgb.png",
    "DL3DV/DL3DV-tables-rgb.png",
    "outdoor/scene_beach2_rgb.png",
]


def _discover_examples() -> list[tuple[str, Path, Path]]:
    examples_root = REPO_ROOT / "examples"
    pairs: list[tuple[str, Path, Path]] = []
    for rel_name in EXAMPLE_ORDER:
        rgb_path = examples_root / rel_name
        if not rgb_path.exists():
            continue

        seg_path = None
        if "_rgb" in rgb_path.name:
            seg_path = rgb_path.with_name(rgb_path.name.replace("_rgb", "_seg"))
        elif "-rgb" in rgb_path.name:
            seg_path = rgb_path.with_name(rgb_path.name.replace("-rgb", "-seg"))
        if seg_path is None or not seg_path.exists():
            continue

        rel = rgb_path.relative_to(examples_root)
        case_name = rgb_path.stem.replace("_rgb", "").replace("-rgb", "")
        label = f"{rel.parent.as_posix()} / {case_name}"
        pairs.append((label, rgb_path, seg_path))
    return pairs


EXAMPLES = _discover_examples()
EXAMPLE_ROWS = [[str(rgb), str(mask)] for _, rgb, mask in EXAMPLES]


@dataclass
class DemoRunState:
    rgb_path: str
    mask_path: str
    output_dir: str
    seed: int
    simplify: float


_sam_cache: dict[str, tuple[AutoProcessor, AutoModelForMaskGeneration]] = {}
_inferencer_cache: dict[tuple[str, str], ISceneInferencer] = {}


def _make_session_dir(request: gr.Request | None, root: Path = UPLOAD_ROOT) -> Path:
    session_hash = getattr(request, "session_hash", None) or uuid.uuid4().hex[:10]
    path = root / session_hash
    path.mkdir(parents=True, exist_ok=True)
    return path


def _timestamped_output_dir(request: gr.Request | None) -> Path:
    session_hash = getattr(request, "session_hash", None) or uuid.uuid4().hex[:10]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return DEFAULT_OUTPUT_ROOT / f"{timestamp}_{session_hash}"


def _get_prompt_image(image_prompts: Any) -> Image.Image | None:
    if image_prompts is None:
        return None
    if isinstance(image_prompts, dict):
        image = image_prompts.get("image")
    else:
        image = image_prompts
    if image is None:
        return None
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    return Image.open(image).convert("RGB")


def _save_prompt_rgb(image_prompts: Any, request: gr.Request | None) -> Path:
    image = _get_prompt_image(image_prompts)
    if image is None:
        raise gr.Error("Please upload an RGB image.")
    session_dir = _make_session_dir(request)
    path = session_dir / "input_rgb.png"
    image.save(path)
    return path


def _resolve_mask_path(mask_path: str | None) -> Path:
    if not mask_path:
        raise gr.Error("Please choose an example or run SAM segmentation first.")
    path = Path(mask_path)
    if not path.exists():
        raise gr.Error(f"Mask file does not exist: {path}")
    return path


def _get_inferencer() -> ISceneInferencer:
    key = (MODEL_ID, BASE_MODEL_ID or "")
    if key not in _inferencer_cache:
        _inferencer_cache[key] = ISceneInferencer.from_pretrained(MODEL_ID, base_model_id=BASE_MODEL_ID)
    return _inferencer_cache[key]


def _get_sam_model(model_choice: str) -> tuple[AutoProcessor, AutoModelForMaskGeneration]:
    model_id = SAM_MODELS[model_choice]
    if model_id in _sam_cache:
        return _sam_cache[model_id]
    processor = AutoProcessor.from_pretrained(model_id)
    segmentator = AutoModelForMaskGeneration.from_pretrained(model_id).to(DEVICE, DTYPE)
    segmentator.eval()
    _sam_cache[model_id] = (processor, segmentator)
    return processor, segmentator


def _boxes_from_prompts(image_prompts: Any) -> list[list[list[int]]]:
    points = image_prompts.get("points", []) if isinstance(image_prompts, dict) else []
    if not points:
        raise gr.Error("Please draw at least one box before running SAM segmentation.")
    boxes = []
    for box in points:
        x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[3]), int(box[4])
        x_min, x_max = sorted((x1, x2))
        y_min, y_max = sorted((y1, y2))
        if x_max <= x_min or y_max <= y_min:
            continue
        boxes.append([x_min, y_min, x_max, y_max])
    if not boxes:
        raise gr.Error("No valid boxes were drawn.")
    return [boxes]


def _mask_to_polygon(mask: np.ndarray) -> list[list[int]] | None:
    import cv2

    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    return contour.reshape(-1, 2).tolist()


def _polygon_to_mask(polygon: list[list[int]], image_shape: tuple[int, int]) -> np.ndarray:
    import cv2

    mask = np.zeros(image_shape, dtype=np.uint8)
    cv2.fillPoly(mask, [np.array(polygon, dtype=np.int32)], color=(1,))
    return mask


def _refine_masks(
    masks: torch.Tensor,
    *,
    polygon_refinement: bool,
    mask_threshold: float,
) -> list[np.ndarray]:
    masks = masks.detach().cpu().float()
    if masks.ndim == 5:
        masks = masks[:, :, 0]
    if masks.ndim == 4:
        masks = masks.mean(dim=1)
    masks = (masks > mask_threshold).numpy().astype(np.uint8)
    refined = [mask for mask in masks]
    if polygon_refinement:
        for idx, mask in enumerate(refined):
            polygon = _mask_to_polygon(mask)
            if polygon is not None:
                refined[idx] = _polygon_to_mask(polygon, mask.shape)
    return refined


def _palette() -> list[int]:
    colors = [0, 0, 0]
    hue = 0.0
    golden_ratio = 0.618033988749895
    for _ in range(1, 256):
        hue = (hue + golden_ratio) % 1.0
        h = hue * 6.0
        c = 0.81
        x = c * (1 - abs(h % 2 - 1))
        m = 0.09
        if h < 1:
            r, g, b = c, x, 0
        elif h < 2:
            r, g, b = x, c, 0
        elif h < 3:
            r, g, b = 0, c, x
        elif h < 4:
            r, g, b = 0, x, c
        elif h < 5:
            r, g, b = x, 0, c
        else:
            r, g, b = c, 0, x
        colors.extend([int((r + m) * 255), int((g + m) * 255), int((b + m) * 255)])
    return colors


def _label_mask_to_pil(label_map: np.ndarray) -> Image.Image:
    if label_map.max(initial=0) < 256:
        image = Image.fromarray(label_map.astype(np.uint8), mode="P")
        image.putpalette(_palette())
        return image
    encoded = np.zeros((*label_map.shape, 3), dtype=np.uint8)
    encoded[..., 0] = label_map & 255
    encoded[..., 1] = (label_map >> 8) & 255
    return Image.fromarray(encoded, mode="RGB")


def resize_prompt_image(image_prompts: Any) -> Any:
    image = _get_prompt_image(image_prompts)
    if image is None:
        return image_prompts
    resized = image.resize(TARGET_SIZE, Image.Resampling.LANCZOS)
    UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)
    path = UPLOAD_ROOT / f"prompt_{uuid.uuid4().hex[:10]}.png"
    resized.save(path)
    return {"image": str(path), "points": []}


def reset_uploaded_image(image_prompts: Any) -> tuple[Any, None, str]:
    return resize_prompt_image(image_prompts), None, ""


def _coerce_file_path(value: Any) -> str:
    if isinstance(value, dict):
        return str(value.get("path") or value.get("name") or value.get("image") or "")
    return str(value or "")


def _raw_example_mask_path(mask_path: Any) -> str:
    selected_mask = Path(_coerce_file_path(mask_path)).name
    for _, _rgb_path, raw_mask_path in EXAMPLES:
        if raw_mask_path.name == selected_mask:
            return str(raw_mask_path)
    return _coerce_file_path(mask_path)


def load_example_pair(rgb_path: Any, mask_path: Any) -> tuple[dict[str, Any], str, str]:
    rgb_value = _coerce_file_path(rgb_path)
    mask_value = _coerce_file_path(mask_path)
    return {"image": rgb_value, "points": []}, mask_value, _raw_example_mask_path(mask_path)


@torch.no_grad()
def run_segmentation(
    image_prompts: Any,
    model_choice: str,
    polygon_refinement: bool,
    mask_threshold: float,
    request: gr.Request,
) -> tuple[str, str]:
    image = _get_prompt_image(image_prompts)
    if image is None:
        raise gr.Error("Please upload an RGB image before running segmentation.")
    boxes = _boxes_from_prompts(image_prompts)
    processor, segmentator = _get_sam_model(model_choice)
    inputs = processor(images=image, input_boxes=boxes, return_tensors="pt").to(segmentator.device, segmentator.dtype)
    outputs = segmentator(**inputs)
    masks = processor.post_process_masks(
        masks=outputs.pred_masks,
        original_sizes=inputs.original_sizes,
        reshaped_input_sizes=inputs.reshaped_input_sizes,
    )[0]
    masks = _refine_masks(masks, polygon_refinement=polygon_refinement, mask_threshold=mask_threshold)

    label_map = np.zeros(image.size[::-1], dtype=np.uint32)
    for idx, mask in enumerate(masks, start=1):
        label_map[mask > 0] = idx

    mask_image = _label_mask_to_pil(label_map)
    session_dir = _make_session_dir(request)
    raw_path = session_dir / "sam_mask.png"
    mask_image.save(raw_path)

    torch.cuda.empty_cache()
    return str(raw_path), str(raw_path)


def run_gaussian_preview(
    image_prompts: Any,
    mask_path: str | None,
    seed: int,
    simplify: float,
    output_dir_text: str,
    request: gr.Request,
) -> tuple[str, dict[str, Any], dict[str, Any], str, DemoRunState]:
    rgb_path = _save_prompt_rgb(image_prompts, request)
    mask_path = _resolve_mask_path(mask_path)
    output_dir = Path(output_dir_text).expanduser() if output_dir_text.strip() else _timestamped_output_dir(request)
    output_dir.mkdir(parents=True, exist_ok=True)

    inferencer = _get_inferencer()
    inferencer.infer_and_save_scene(
        scene_rgb_path=rgb_path,
        instance_seg_path=mask_path,
        output_dir=output_dir,
        overwrite=True,
        save_dbg=False,
        simplify=float(simplify),
        only_3dgs=True,
        seed=int(seed),
    )

    scene_ply = output_dir / "scene_pred.ply"
    if not scene_ply.exists():
        raise gr.Error(f"Generation finished but scene_pred.ply was not found in {output_dir}")

    state = DemoRunState(
        rgb_path=str(rgb_path),
        mask_path=str(mask_path),
        output_dir=str(output_dir),
        seed=int(seed),
        simplify=float(simplify),
    )
    torch.cuda.empty_cache()
    return (
        str(scene_ply),
        gr.update(value=str(scene_ply), interactive=True),
        gr.update(value=None, interactive=False),
        "",
        state,
    )


def _progress_bar(percent: int) -> str:
    percent = max(0, min(100, int(percent)))
    return f"""
    <div style="height: 14px; width: 100%; background: #ece7dc; border-radius: 999px; overflow: hidden; border: 1px solid #d8cbb7;">
      <div style="height: 100%; width: {percent}%; background: linear-gradient(90deg, #b77a2f, #e0b15a); transition: width 0.4s ease;"></div>
    </div>
    """


def run_glb_export(
    state: DemoRunState | dict[str, Any] | None,
    simplify: float,
) -> Any:
    if state is None:
        raise gr.Error("Please run GS preview first so the demo knows which RGB/mask/output directory to use.")
    if isinstance(state, dict):
        state = DemoRunState(**state)

    output_dir = Path(state.output_dir)
    yield gr.update(value=None, interactive=False), _progress_bar(5), gr.update(value=None)
    inferencer = _get_inferencer()
    yield gr.update(value=None, interactive=False), _progress_bar(15), gr.update(value=None)
    inferencer.infer_and_save_scene(
        scene_rgb_path=state.rgb_path,
        instance_seg_path=state.mask_path,
        output_dir=output_dir,
        overwrite=True,
        save_dbg=False,
        simplify=float(simplify),
        only_3dgs=False,
        seed=int(state.seed),
    )

    scene_glb = output_dir / "scene_pred.glb"
    if not scene_glb.exists():
        raise gr.Error(f"GLB export finished but scene_pred.glb was not found in {output_dir}")

    torch.cuda.empty_cache()
    yield gr.update(value=str(scene_glb), interactive=True), _progress_bar(100), str(scene_glb)


def clear_glb_outputs() -> tuple[dict[str, Any], str, None, dict[str, Any]]:
    return gr.update(value=None, interactive=False), "", None, gr.update(value=None)


def clear_generation_outputs() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], str, None, dict[str, Any]]:
    return (
        gr.update(value=None),
        gr.update(value=None, interactive=False),
        gr.update(value=None, interactive=False),
        "",
        None,
        gr.update(value=None),
    )


def build_demo() -> gr.Blocks:
    with gr.Blocks(title="I-Scene Interactive Demo", delete_cache=(3600, 3600)) as demo:
        gr.Markdown(MARKDOWN)

        run_state = gr.State(None)

        with gr.Row():
            with gr.Column(scale=1):
                image_prompts = ImagePrompter(
                    label="RGB image (upload, then optionally draw boxes for SAM)",
                    type="pil",
                    height=520,
                )

                with gr.Row():
                    segment_button = gr.Button("Run SAM Segmentation", variant="secondary")

                with gr.Accordion("Segmentation settings", open=False):
                    sam_model = gr.Dropdown(
                        choices=list(SAM_MODELS.keys()),
                        value="sam-vit-huge (best quality, 636M)",
                        label="SAM model",
                    )
                    mask_threshold = gr.Slider(
                        minimum=-1.0,
                        maximum=1.0,
                        value=0.0,
                        step=0.05,
                        label="Mask threshold",
                    )
                    polygon_refinement = gr.Checkbox(
                        label="Polygon refinement",
                        value=False,
                    )

                sam_mask_preview = gr.Image(
                    label="Instance mask",
                    type="filepath",
                    format="png",
                    height=260,
                )
                mask_path_value = gr.Textbox(visible=False)

                with gr.Accordion("Generation settings", open=False):
                    seed = gr.Number(label="Seed", value=DEFAULT_SEED, precision=0)
                    simplify = gr.Slider(
                        minimum=0.5,
                        maximum=1.0,
                        value=DEFAULT_SIMPLIFY,
                        step=0.01,
                        label="GLB mesh simplify ratio",
                    )
                    output_dir = gr.Textbox(
                        label="Output directory (optional)",
                        placeholder="Leave empty to use outputs/demo/<timestamp>_<session>",
                    )

                generate_gs_button = gr.Button("Generate Gaussian Splatting Preview", variant="primary", size="lg")

            with gr.Column(scale=1):
                preview = LitModel3D(
                    label="3D preview",
                    exposure=10.0,
                    height=520,
                )
                download_gs = gr.DownloadButton(
                    label="Download Gaussian Splatting PLY",
                    interactive=False,
                )

                with gr.Row():
                    generate_glb_button = gr.Button("Generate GLB", variant="secondary")
                glb_progress = gr.HTML(value="")
                glb_preview = gr.Model3D(
                    label="GLB mesh preview",
                    clear_color=(0.98, 0.96, 0.91, 1.0),
                    display_mode="solid",
                    height=360,
                )
                download_glb = gr.DownloadButton(
                    label="Download Mesh GLB",
                    interactive=False,
                )

        image_prompts.upload(
            reset_uploaded_image,
            inputs=[image_prompts],
            outputs=[image_prompts, sam_mask_preview, mask_path_value],
        )

        segment_button.click(
            run_segmentation,
            inputs=[image_prompts, sam_model, polygon_refinement, mask_threshold],
            outputs=[sam_mask_preview, mask_path_value],
        )

        generate_gs_button.click(
            clear_generation_outputs,
            outputs=[preview, download_gs, download_glb, glb_progress, run_state, glb_preview],
            show_progress="hidden",
        ).then(
            run_gaussian_preview,
            inputs=[
                image_prompts,
                mask_path_value,
                seed,
                simplify,
                output_dir,
            ],
            outputs=[preview, download_gs, download_glb, glb_progress, run_state],
            show_progress="full",
        )

        generate_glb_button.click(
            run_glb_export,
            inputs=[run_state, simplify],
            outputs=[download_glb, glb_progress, glb_preview],
            show_progress="hidden",
        )

        example_rgb = gr.Image(label="RGB", type="filepath", visible=False)
        example_mask = gr.Image(label="Instance mask", type="filepath", visible=False)

        with gr.Row():
            gr.Examples(
                examples=EXAMPLE_ROWS,
                inputs=[example_rgb, example_mask],
                outputs=[image_prompts, sam_mask_preview, mask_path_value],
                fn=load_example_pair,
                cache_examples=False,
                label="Examples",
                run_on_click=True,
            )

    return demo


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server_name", default="0.0.0.0")
    parser.add_argument("--server_port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="I-Scene model id or local model package path.")
    parser.add_argument(
        "--base_model",
        default=None,
        help="Optional TRELLIS base model id or local mirror path. Defaults to the model package metadata.",
    )
    return parser.parse_args()


def main() -> None:
    global MODEL_ID, BASE_MODEL_ID

    args = parse_args()
    MODEL_ID = args.model
    BASE_MODEL_ID = args.base_model
    DEFAULT_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)
    demo = build_demo()
    demo.queue()
    demo.launch(
        server_name=args.server_name,
        server_port=args.server_port,
        share=args.share,
    )


if __name__ == "__main__":
    main()
