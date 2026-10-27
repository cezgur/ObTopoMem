# Copyright 2026 Ata Baran Fırat
# SPDX-License-Identifier: Apache-2.0

"""
Example command-line entry

python -m obtopomem.main --video ~/Videos/example.mp4 --output-dir ~/obtopomem_data/example

To resume or visualize a prerecorded run, add --reuse-cache.
Use different output directory whenever input data or processing settings change.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import time
from pathlib import Path
from typing import Optional
import sys

# Defaults

DEFAULT_MAX_DIM = 630
NUM_KEYPOINTS_SP = 4096
KEYPOINT_DENSITY_THRESHOLD = 0.001
KEY_FRAME_INTERVAL_SECONDS = 0.1

SAM_PRED_IOU_THRESH = 0.95
SAM_STABILITY_SCORE_THRESH = 0.98
SAM_BOX_NMS_THRESH = 0.2
SAM_CROP_NMS_THRESH = 0.05

_MANIFEST_NAME = "run_config.json"
_LOG = logging.getLogger(__name__)

def _make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__,
                                     allow_abbrev=False)

    parser.add_argument("--video", type=Path, help="Input video path.")

    parser.add_argument(
        "--output-dir", required=True, type=Path,
        help="Directory for caches and visualizations (use a new directory for fresh run)."
    )

    parser.add_argument(
        "--reuse-cache", action="store_true",
        help="Resume/reuse an existing run only if its recorded inputs and settings match."
    )

    parser.add_argument(
        "--use-cached-config", action="store_true",
        help="Load video path and processing settings from the output directory's run_config.json."
    )

    parser.add_argument(
        "--key-frame-interval-seconds", type=float, default=KEY_FRAME_INTERVAL_SECONDS,
        help="Keyframe interval in seconds; -1 selects every entry in frame (default: %(default)s)"
    )

    parser.add_argument(
        "--max-keypoints", type=int, default=NUM_KEYPOINTS_SP,
        help="Maximum SuperPoint keypoints; 0 means no cap (default: %(default)s)."
    )

    parser.add_argument(
        "--max-dim", type=int, default=DEFAULT_MAX_DIM,
        help="Starting image dimension for feature extraction/calibration."
    )

    parser.add_argument(
        "--keypoint-density-threshold", type=float,
        default=KEYPOINT_DENSITY_THRESHOLD
    )

    parser.add_argument(
        "--no-calibrate-vram", action="store_false", dest="calibrate_vram",
        help="Skip feature-extractor VRAM calibration and use --max-dim directly."
    )

    parser.add_argument("--sam-pred-iou-thresh", type=float, default=SAM_PRED_IOU_THRESH)

    parser.add_argument("--sam-stability-score-thresh", type=float, default=SAM_STABILITY_SCORE_THRESH)

    parser.add_argument("--sam-box-nms-thresh", type=float, default=SAM_BOX_NMS_THRESH)

    parser.add_argument("--sam-crop-nms-thresh", type=float, default=SAM_CROP_NMS_THRESH)

    return parser

def _validate_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> Path:
    if args.video is None:
        parser.error("--video is required unless --use-cached-config is specified.")
    video = args.video.expanduser().resolve()

    if not video.is_file():
        parser.error(f"Input video is not an existing file: {video}.")

    interval = args.key_frame_interval_seconds
    if not math.isfinite(interval) or (interval <= 0 and interval != -1):
        parser.error("--key-frame-interval-seconds must be positive or -1.")
    if args.max_dim <= 0 or (args.calibrate_vram and args.max_dim < 512):
        parser.error("--max-dim must be positive (at least 512 with VRAM calibration).")
    if args.max_keypoints < 0:
        parser.error("--max-keypoints must be >= 0.")
    if not math.isfinite(args.keypoint_density_threshold) or args.keypoint_density_threshold < 0:
        parser.error("--keypoint-density-threshold must be finite and >= 0.")
    for option in (
        "sam_pred_iou_thresh", "sam_stability_score_thresh",
        "sam_box_nms_thresh", "sam_crop_nms_thresh"
    ):
        value = getattr(args, option)
        if not math.isfinite(value) or not 0 <= value <= 1:
            parser.error(f"--{option.replace('_', '-')} must be between 0 and 1.")
    return video

def _check_models() -> None:
    """
    Validate only checkpoint files/exports; full loading happen in the pipeline.
    """
    from .paths import OMNIGLUE_MODEL_DIR, SAM_MODEL_DIR

    required = {
        "OmniGlue": OMNIGLUE_MODEL_DIR / "og_export" / "saved_model.pb",
        "SuperPoint": OMNIGLUE_MODEL_DIR / "sp_v6" / "saved_model.pb",
        "DINOv2": OMNIGLUE_MODEL_DIR / "dinov2_vitb14_pretrain.pth",
        "SAM ViT-L": SAM_MODEL_DIR / "sam_vit_l_0b3195.pth"
    }
    missing = [f"{name}: {path}" for name, path in required.items()
               if not path.is_file() or path.stat().st_size == 0]

    if missing:
        raise FileNotFoundError(
            "Missing model checkpoints/exports:\n " + "\n ".join(missing)
            + "\nInstall them with: python scripts/download_models.py"
        )

def _run_manifest(video: Path, args: argparse.Namespace) -> dict:
    stat = video.stat()
    return {
        "format_version": 1,
        "video": {
            "path": str(video),
            "size_bytes": stat.st_size,
            "mtime_ns": stat.st_mtime_ns
        },
        "processing": {
            "calibrate_vram": args.calibrate_vram,
            "max_dim": args.max_dim,
            "max_keypoints": args.max_keypoints,
            "keypoint_density_threshold": args.keypoint_density_threshold,
            "key_frame_interval_seconds": args.key_frame_interval_seconds,
            "use_roi": False,
            "sam_model_type": "vit_l",
            "sam_pred_iou_thresh": args.sam_pred_iou_thresh,
            "sam_stability_score_thresh": args.sam_stability_score_thresh,
            "sam_box_nms_thresh": args.sam_box_nms_thresh,
            "sam_crop_nms_thresh": args.sam_crop_nms_thresh
        }
    }

def _load_cached_config(output_dir: Path,
                        parser: argparse.ArgumentParser) -> dict:
    """
    Load and validate a previously saved run configuration.
    """
    manifest_path = output_dir / _MANIFEST_NAME

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        parser.error(f"No cached configuration found at: {manifest_path}.")
    except (OSError, json.JSONDecodeError) as e:
        parser.error(f"Could not read cached configuration: {e}")

    if (not isinstance(manifest, dict) or
        type(manifest.get("format_version")) is not int or
        manifest.get("format_version") != 1 or
        not isinstance(manifest.get("video"), dict) or
        not isinstance(manifest.get("processing"), dict)):

        parser.error(f"Invalid or unsupported cached configuration: {manifest_path}")

    video_path = manifest["video"].get("path")

    if (not isinstance(video_path, str) or
        not video_path.strip() or
        Path(video_path) == Path(".")):
        parser.error(f"Invalid video path in cached configuration.")

    processing = manifest["processing"]

    if type(processing.get("calibrate_vram")) is not bool:
        parser.error("calibrate_vram must be a boolean.")

    for key in ("max_dim", "max_keypoints"):
        if type(processing.get(key)) is not int:
            parser.error(f"{key} must be an integer.")

    numeric_parameters = ("keypoint_density_threshold",
                          "key_frame_interval_seconds",
                          "sam_pred_iou_thresh",
                          "sam_stability_score_thresh",
                          "sam_box_nms_thresh",
                          "sam_crop_nms_thresh")

    for key in numeric_parameters:
        value = processing.get(key)

        if type(value) not in (int, float) or not math.isfinite(value):
            parser.error(f"{key} must be a finite number.")

    return manifest

def _prepare_output_dir(output_dir: Path, expected: dict, reuse_cache: bool) -> None:
    """
    Refuse to overwrite unknown or incompatible cached results.
    """
    manifest_path = output_dir / _MANIFEST_NAME
    if output_dir.exists():
        if not output_dir.is_dir():
            raise ValueError(f"Output path is not a directory: {output_dir}.")
        if manifest_path.is_file():
            try:
                actual = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (ValueError, OSError) as exc:
                raise RuntimeError(f"Invalid run manifest: {manifest_path}") from exc
            if actual != expected:
                raise RuntimeError(
                    f"Output directory belongs to a different input or configuration: {output_dir}\n"
                    "Choose a new --output-dir; existing caches have not been verified."
                )
            if not reuse_cache:
                raise RuntimeError(
                    f"Output directory already contains a recorded run: {output_dir}\n"
                    "Use --reuse-cache to continue it, or choose a new --output-dir."
                )

            _LOG.info("Compatible run configuration found; cache reuse is enabled.")
            return
        if any(output_dir.iterdir()):
            raise RuntimeError(
                f" Output directory is not empty and has no {_MANIFEST_NAME}: {output_dir}\n"
                "Use a new empty --output-dir; existing files will not be deleted."
            )
    else:
        output_dir.mkdir(parents=True, exist_ok=True)

    # Write before processing so an interrupted run cannot silently mix settings.
    temporary = output_dir / f".{_MANIFEST_NAME}.{os.getpid()}.tmp"
    try:
        temporary.write_text(json.dumps(expected, indent=2, sort_keys=True) + "\n",
                             encoding="utf-8")
        os.replace(temporary, manifest_path)
    finally:
        temporary.unlink(missing_ok=True)


def main(argv: Optional[list[str]] = None) -> int:
    parser = _make_parser()

    raw_args = sys.argv[1:] if argv is None else argv
    args = parser.parse_args(raw_args)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    output_dir = args.output_dir.expanduser().resolve()

    if args.use_cached_config:
        processing_options = {
            "--video",
            "--max-dim",
            "--max-keypoints",
            "--keypoint-density-threshold",
            "--key-frame-interval-seconds",
            "--no-calibrate-vram",
            "--sam-pred-iou-thresh",
            "--sam-stability-score-thresh",
            "--sam-box-nms-thresh",
            "--sam-crop-nms-thresh",
        }

        supplied_options = {
            arg.split("=", 1)[0]
            for arg in raw_args
            if arg.startswith("--")
        }

        conflicts = processing_options & supplied_options

        if conflicts:
            parser.error(
                "--use-cached-config cannot be combined with: "
                + ", ".join(sorted(conflicts))
            )

        # Read the existing run configuration.
        cached = _load_cached_config(output_dir, parser)

        video_config = cached["video"]
        processing = cached["processing"]

        try:
            args.video = Path(video_config["path"])

            args.calibrate_vram = processing["calibrate_vram"]
            args.max_dim = processing["max_dim"]
            args.max_keypoints = processing["max_keypoints"]
            args.keypoint_density_threshold = processing["keypoint_density_threshold"]
            args.key_frame_interval_seconds = processing["key_frame_interval_seconds"]

            args.sam_pred_iou_thresh = processing["sam_pred_iou_thresh"]
            args.sam_stability_score_thresh = processing["sam_stability_score_thresh"]
            args.sam_box_nms_thresh = processing["sam_box_nms_thresh"]
            args.sam_crop_nms_thresh = processing["sam_crop_nms_thresh"]

        except (KeyError, TypeError, ValueError) as exc:
            parser.error(f"Incomplete or invalid cached configuration: {exc}")

        # Cached configuration mode always reuses the existing run.
        args.reuse_cache = True

    # Existing validation and setup.
    video = _validate_args(args, parser)

    # Validate checkpoints before loading heavy dependencies.
    _check_models()

    manifest = _run_manifest(video, args)

    _prepare_output_dir(
        output_dir,
        manifest,
        args.reuse_cache
    )

    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

    from .TopoEnv import TopoEnvCacheManager
    from .VideoProcessor import VideoProcessor
    from .utils import visualize_video

    started = time.perf_counter()
    processor = None
    topoenv = None
    try:
        env_id = f"{video.stem}_TopoEnv"
        topoenv_manager = TopoEnvCacheManager(str(output_dir))
        topoenv = topoenv_manager.get(env_id, optional=True)

        if topoenv is None:
            _LOG.info(f"No cached TopoEnv found; processing {video}.")
            processor = VideoProcessor(
                video_path=str(video),
                cache_dir=str(output_dir),
                calibrate_vram=args.calibrate_vram,
                default_max_dim=args.max_dim,
                use_roi=False,
                max_keypoints=args.max_keypoints,
                keypoint_density_threshold=args.keypoint_density_threshold,
                sam_pred_iou_thresh=args.sam_pred_iou_thresh,
                sam_stability_score_thresh=args.sam_stability_score_thresh,
                sam_box_nms_thresh=args.sam_box_nms_thresh,
                sam_crop_nms_thresh=args.sam_crop_nms_thresh
            )
            topoenv = processor.process_video(args.key_frame_interval_seconds)
            filled, _, _ = topoenv.topoManFilled_w_log
            _LOG.info(f"topoManFilled shape: {filled.shape}")

            processor.topoenv_manager.evict_all()
        else:
            _LOG.info(f"Loaded TopoEnv from cache: {env_id}")

        visualize_video(topoenv=topoenv, cache_dir=output_dir)
        _LOG.info(f"Completed in {time.perf_counter() - started:.1f} seconds.")

        return 0

    finally:
        if topoenv is not None:
            topoenv.videoContainer.release()
        if processor is not None and (
            topoenv is None or processor.videoContainer is not topoenv.videoContainer
        ):
            processor.videoContainer.release()


if __name__ == "__main__":
    raise SystemExit(main())
