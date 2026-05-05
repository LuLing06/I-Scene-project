"""Run single-sample I-Scene inference."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


DEFAULT_MODEL = "LuLing/IScene"
DEFAULT_SEED = 43
DEFAULT_SIMPLIFY = 0.95


def bootstrap_imports() -> None:
    repo_root = Path(__file__).resolve().parent

    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="I-Scene model id or local model package path.",
    )
    parser.add_argument(
        "--base_model",
        default=None,
        help="Optional TRELLIS base model id or local mirror path. Defaults to the model package metadata.",
    )
    parser.add_argument("--rgb", type=Path, required=True, help="Path to the input RGB image.")
    parser.add_argument("--mask", type=Path, required=True, help="Path to the instance segmentation mask.")
    parser.add_argument("--output_dir", type=Path, required=True, help="Directory for generated outputs.")
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Random seed for inference.",
    )
    parser.add_argument(
        "--simplify",
        type=float,
        default=DEFAULT_SIMPLIFY,
        help="Mesh simplification ratio for GLB export.",
    )
    parser.add_argument(
        "--save_dbg",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Save debug artifacts alongside the release outputs.",
    )
    parser.add_argument(
        "--only_3dgs",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Skip GLB export and only save Gaussian assets.",
    )
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if not args.rgb.exists():
        raise FileNotFoundError(f"RGB file does not exist: {args.rgb}")
    if not args.mask.exists():
        raise FileNotFoundError(f"Mask file does not exist: {args.mask}")


def main(argv: list[str] | None = None) -> int:
    bootstrap_imports()
    parser = build_parser()
    args = parser.parse_args(argv)
    validate_args(args)

    from iscene.inference.inferencer import ISceneInferencer

    inferencer = ISceneInferencer.from_pretrained(args.model, base_model_id=args.base_model)
    inferencer.infer_and_save_scene(
        scene_rgb_path=args.rgb,
        instance_seg_path=args.mask,
        output_dir=args.output_dir,
        overwrite=True,
        save_dbg=args.save_dbg,
        simplify=args.simplify,
        only_3dgs=args.only_3dgs,
        seed=args.seed,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
