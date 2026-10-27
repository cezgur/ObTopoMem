# Copyright 2026 Ata Baran Fırat
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

from pathlib import Path
from tqdm import tqdm
from typing import List, Optional, Dict, Tuple

import cv2
import numpy as np
import re

import matplotlib.pyplot as plt

import ffmpeg
import subprocess

from .FrameDescriptorBase import FrameDescriptorBase
from .FrameSegmentDescriptor import SegmentGraph
from .AnyThing import AnyThing, AnyThingGraph
from .TopoEnv import TopoEnv


def draw_epipolar_lines(topoenv: TopoEnv,
                        Fmap: dict,
                        F_sampson: dict,
                        anything_colors: dict,
                        cache_dir: Path):
    logger.info("Generating epipolar line frames.")
    EPIPOLAR_LINE_FRAMES_DIR = cache_dir / "epipolar_line_frames"
    EPIPOLAR_LINE_FRAMES_DIR.mkdir(exist_ok=True)

    KEYPOINT_FRAMES_DIR = cache_dir / "keypoint_frames"

    valid_frame_ids = set(topoenv.frame_ids)
    valid_frame_indices = [
        idx for idx in range(len(topoenv.videoContainer))
        if FrameDescriptorBase.create_id(topoenv.video_id, idx) in valid_frame_ids
    ]

    all_sampson_errors = np.concatenate(list(F_sampson.values())) if F_sampson else np.array([1e-5])
    sampson_error_scale = np.percentile(all_sampson_errors, 95)
    if sampson_error_scale == 0:
        sampson_error_scale = 1e-5

    for anything_id in tqdm(topoenv.anything_ids, desc="Generating epipolar line frames for each anything."):
        anything_output_dir = EPIPOLAR_LINE_FRAMES_DIR / anything_id
        anything_output_dir.mkdir(exist_ok=True)

        for frame_idx_j_chrono in valid_frame_indices:
            keypoint_frame_path = KEYPOINT_FRAMES_DIR / f"frame_{frame_idx_j_chrono:04d}.png"
            if not keypoint_frame_path.exists():
                height, width = topoenv.videoContainer.height, topoenv.videoContainer.width
                frame_j_image = np.zeros((height, width, 3), dtype=np.uint8)
            else:
                frame_j_image = cv2.imread(str(keypoint_frame_path))

            frame_j_id = FrameDescriptorBase.create_id(topoenv.video_id, frame_idx_j_chrono)
            topoenv_frame_idx_j = topoenv.frame_id_index(frame_j_id)  # Get internal index

            for frame_idx_i_chrono in valid_frame_indices:
                if frame_idx_i_chrono == frame_idx_j_chrono:
                    continue

                frame_i_id = FrameDescriptorBase.create_id(topoenv.video_id, frame_idx_i_chrono)
                topoenv_frame_idx_i = topoenv.frame_id_index(frame_i_id)  # Get internal index

                # Use internal indices for Fmap access
                pair = (
                    (topoenv_frame_idx_i, topoenv_frame_idx_j)
                    if topoenv_frame_idx_i < topoenv_frame_idx_j
                    else (topoenv_frame_idx_j, topoenv_frame_idx_i)
                )
                F = Fmap.get(pair)
                if F is None:
                    continue

                if topoenv_frame_idx_i > topoenv_frame_idx_j:
                    F = F.T

                anything = topoenv.anything_manager.get(anything_id)
                if anything and frame_i_id in anything.frame_ids:
                    center_pix = anything.center_in_frames([frame_i_id])[0]
                    if center_pix is None or np.isnan(center_pix).any():
                        continue

                    x_i_h = np.array([center_pix[0], center_pix[1], 1.0])

                    l_j = F @ x_i_h
                    a, b, c = l_j

                    color_rgb = anything_colors.get(anything_id, (255, 255, 255))
                    color_bgr = color_rgb # Use RGB directly, cv2.line will interpret as BGR
                    height, width, _ = frame_j_image.shape

                    if abs(b) > abs(a):
                        x0 = 0
                        y0 = int(-c / b)
                        x1 = width - 1
                        y1 = int(-(a * x1 + c) / b)
                    else:
                        y0 = 0
                        x0 = int(-c / a)
                        y1 = height - 1
                        x1 = int(-(b * y1 + c) / a)

                    sampson_errors = F_sampson.get(pair)
                    if sampson_errors is not None:
                        mean_error = np.mean(sampson_errors)
                        # Normalize the error and compute alpha. Alpha is inverse to the error.
                        alpha_line = 1.0 - np.clip(mean_error / sampson_error_scale, 0, 1)

                        # Create a temporary image for the current line
                        single_line_image = np.zeros_like(frame_j_image, dtype=np.uint8)
                        cv2.line(single_line_image, (x0, y0), (x1, y1), color_bgr, 1)

                        # Blend the single line with the frame_j_image
                        frame_j_image = cv2.addWeighted(single_line_image, alpha_line, frame_j_image, 1 - alpha_line, 0)

            save_path = anything_output_dir / f"frame_{frame_idx_j_chrono:04d}.png"
            cv2.imwrite(str(save_path), frame_j_image)

def plot_anything_lifespans(
    topoenv: TopoEnv,
    output_dir: Path,
    anything_colors: Dict[str, Tuple[int, int, int]],
    frame_indices: Optional[List[int]] = None,
):
    """
    Visualize AnyThing lifespans as filled horizontal blocks over frame indices,
    plus representative thumbnails on the right.

    If `frame_indices` is provided, only those frames are shown. Contiguity is
    defined in the *display index* space (consecutive entries in the filtered
    index list form a continuous block).
    """
    logger.info("Generating AnyThing lifespan plot...")

    anything_objects = [topoenv.anything_manager.get(at_id) for at_id in topoenv.anything_ids]
    anything_objects = [at for at in anything_objects if at is not None]

    if not anything_objects:
        logger.info("No AnyThing objects to plot.")
        return

    total_frames = len(topoenv.videoContainer)

    # --- Decide which frame indices to display ---
    if frame_indices is None:
        used_frame_indices = list(range(total_frames))
    else:
        used_frame_indices = sorted({idx for idx in frame_indices if 0 <= idx < total_frames})

    if not used_frame_indices:
        logger.info("No valid frame indices provided for plotting.")
        return

    # Map original frame index -> display index (0..n_frames_disp-1)
    frame_idx_to_disp_idx: Dict[int, int] = {
        orig_idx: disp_idx for disp_idx, orig_idx in enumerate(used_frame_indices)
    }
    n_frames_disp = len(used_frame_indices)
    num_anythings = len(anything_objects)

    fig, ax = plt.subplots(figsize=(24, 12))

    # --- Draw lifespans as continuous horizontal blocks ---
    for i, anything in enumerate(anything_objects):
        disp_indices: List[int] = []

        # Collect display indices from frame_ids, respecting the filtered frames
        for fid in anything.frame_ids:
            m = re.search(r"\d+$", fid)
            if m:
                orig_idx = int(m.group(0))
                if orig_idx in frame_idx_to_disp_idx:
                    disp_indices.append(frame_idx_to_disp_idx[orig_idx])

        disp_indices.sort()
        if not disp_indices:
            continue

        # Group into contiguous ranges in display index space
        ranges: List[Tuple[int, int]] = []
        start = disp_indices[0]
        prev = disp_indices[0]
        for idx in disp_indices[1:]:
            if idx == prev + 1:
                prev = idx
            else:
                ranges.append((start, prev - start + 1))
                start = idx
                prev = idx
        ranges.append((start, prev - start + 1))

        color = np.array(anything_colors.get(anything.id, (255, 255, 255)), dtype=float) / 255.0

        ax.broken_barh(
            ranges,
            (i - 0.4, 0.8),  # vertical span of this AnyThing row
            facecolors=[color],
            edgecolors="none",
        )

    # --- Axes styling ---
    ax.set_xlim(-0.5, n_frames_disp - 0.5)
    ax.set_ylim(-0.5, num_anythings - 0.5)
    ax.set_xlabel("Frame Index")
    ax.set_ylabel("")          # no random anything_ids
    ax.set_yticks([])          # hide y tick labels completely
    ax.set_title("AnyThing Lifespans Across Video Frames")

    # Show original frame numbers as x tick labels (downsample if many)
    max_ticks = 20
    if n_frames_disp <= max_ticks:
        tick_positions = list(range(n_frames_disp))
    else:
        step = max(1, n_frames_disp // max_ticks)
        tick_positions = list(range(0, n_frames_disp, step))
    tick_labels = [str(used_frame_indices[pos]) for pos in tick_positions]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=45, ha="right")

    ax.grid(True, axis="x", linestyle="--", alpha=0.3)

    # --- Layout for thumbnails on the right (small gap, no huge empty area) ---
    REPRESENTATIVE_IMAGE_SIZE_PX = 50   # just for aspect; height is data-driven
    NUM_REPR_IMAGES = 5

    # Reserve a narrow vertical strip on the right for thumbnails
    right_margin = 0.82  # fraction of figure width occupied by the main plot
    plt.subplots_adjust(right=right_margin)

    # Get updated axis position and y-limits
    ax_pos = ax.get_position()  # Bbox in figure fraction
    y_min, y_max = ax.get_ylim()
    y_range = y_max - y_min

    pad_x = 0.01
    thumb_area_left = right_margin + pad_x
    thumb_area_right = 1.0 - pad_x
    thumb_area_width = max(thumb_area_right - thumb_area_left, 1e-3)

    thumb_width_fig = thumb_area_width / NUM_REPR_IMAGES

    # Height of one row (in figure coords) based on the 0.8 data-height used
    row_data_height = 0.8
    row_fig_height = (row_data_height / y_range) * ax_pos.height
    thumb_height_fig = row_fig_height * 0.9  # a bit smaller to avoid touching borders

    for i, anything in enumerate(anything_objects):
        repr_seg_ids = anything.representative_segment_ids[:NUM_REPR_IMAGES]

        # y center of this row in data coordinates is i
        y_center_data = i
        # Convert to figure coordinates
        frac_in_axis = (y_center_data - y_min) / y_range
        y_fig_center = ax_pos.y0 + frac_in_axis * ax_pos.height
        y_fig_bottom = y_fig_center - thumb_height_fig / 2

        for j, seg_id in enumerate(repr_seg_ids):
            segment = topoenv.segment_manager.get(seg_id)
            if not segment:
                continue

            original_frame = topoenv.videoContainer[segment.frame_idx]  # RGB
            x, y, w, h = segment._roi_bbox

            img_h_orig, img_w_orig, _ = original_frame.shape
            x = max(0, x)
            y = max(0, y)
            w = min(w, img_w_orig - x)
            h = min(h, img_h_orig - y)

            if w <= 0 or h <= 0:
                continue

            cropped_segment = original_frame[y : y + h, x : x + w]  # RGB
            resized_segment = cv2.resize(
                cropped_segment,
                (REPRESENTATIVE_IMAGE_SIZE_PX, REPRESENTATIVE_IMAGE_SIZE_PX),
            )

            x_pos_fig = thumb_area_left + j * thumb_width_fig

            img_ax = fig.add_axes((x_pos_fig, y_fig_bottom, thumb_width_fig, thumb_height_fig))
            img_ax.imshow(resized_segment)  # already RGB
            img_ax.axis("off")

    output_path = output_dir / "anything_lifespans.png"
    plt.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    logger.info(f"AnyThing lifespan plot saved to {output_path}")

def visualize_video(topoenv: TopoEnv,
                    cache_dir: Path):
    logger.info("Starting video visualization.")

    output_video_path = cache_dir / "segment_wsam_output.mp4"

    # --- Save representative segments ---
    logger.info("Saving representative segments as PNG files.")
    REPRESENTATIVE_SEGMENTS_DIR = cache_dir / "representative_segments"
    REPRESENTATIVE_SEGMENTS_DIR.mkdir(exist_ok=True)

    for anything_id in tqdm(topoenv.anything_ids, desc="Saving representative segments"):
        anything = topoenv.anything_manager.get(anything_id, optional=False)
        assert anything is not None

        anything_output_dir = REPRESENTATIVE_SEGMENTS_DIR / anything_id
        anything_output_dir.mkdir(exist_ok=True)

        for seg_id in anything.representative_segment_ids:
            segment = topoenv.segment_manager.get(seg_id, optional=False)
            assert segment is not None

            frame_idx = segment.frame_idx
            original_frame = topoenv.videoContainer[frame_idx]

            x, y, w, h = segment._roi_bbox

            # Ensure bounding box coordinates are within frame boundaries
            img_h, img_w, _ = original_frame.shape
            x = max(0, x)
            y = max(0, y)
            w = min(w, img_w - x)
            h = min(h, img_h - y)

            if w > 0 and h > 0:
                cropped_segment = original_frame[y : y + h, x : x + w]
                save_path = anything_output_dir / f"{seg_id}.png"
                cv2.imwrite(str(save_path), cv2.cvtColor(cropped_segment, cv2.COLOR_RGB2BGR))
            else:
                logger.warning(
                    f"Skipping saving segment {seg_id} for AnyThing {anything_id} due to invalid "
                    f"bounding box: (x={x}, y={y}, w={w}, h={h})"
                )

    fps = topoenv.videoContainer.fps
    width = topoenv.videoContainer.width
    height = topoenv.videoContainer.height
    ffmpeg_log_path = cache_dir / "ffmpeg_export.log"
    try:
        ffmpeg_stream = (
            ffmpeg.input(
                "pipe:",
                format="rawvideo",
                pix_fmt="rgb24",
                s=f"{width}x{height}",
                r=fps,
            ).output(
                str(output_video_path),
                pix_fmt="yuv420p",
                vcodec="h264_nvenc",
            ).overwrite_output()
        )

        with ffmpeg_log_path.open("wb") as ffmpeg_log:
            process = subprocess.Popen(
                ffmpeg.compile(ffmpeg_stream),
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=ffmpeg_log
            )
    except (ffmpeg.Error, OSError) as e:
        logger.error(f"Could not start FFmpeg process: {e}")
        process = None

    if process is None:
        raise RuntimeError(
            "Video export failed: FFmpeg could not be started. "
            "verify that FFmpeg is installed and available on PATH."
        )
    else:
        stdin = process.stdin

        if stdin is None:
            process.kill()
            process.wait()
            raise RuntimeError("FFmpeg stdin pipe was not created.")

        try:
            # Retrieve all AnyThing objects and assign them a consistent color.
            segment_ids_per_anything: List[List[str]] = []
            anything_ids = topoenv.anything_ids
            for anything_id in topoenv.anything_ids:
                at = topoenv.anything_manager.get(anything_id, optional=False)
                assert at is not None
                segment_ids_per_anything.append(at.segment_ids)
            all_segment_ids: List[str] = []
            [all_segment_ids.extend(segment_ids_single_anything) for segment_ids_single_anything in segment_ids_per_anything]
            sg = SegmentGraph(
                segment_ids=all_segment_ids,
                segment_manager=topoenv.segment_manager,
                matches_manager=topoenv.matches_manager
            )
            atg = AnyThingGraph(
                anything_ids=anything_ids,
                anything_manager=topoenv.anything_manager,
                matches_manager=topoenv.matches_manager,
                segment_manager=topoenv.segment_manager,
                segment_graph=sg,
                cache_dir=str(cache_dir)
            )
            atg.visualize_graph_view(save_folder=str(cache_dir / "anything_graph.png"))
            sg.visualize_graph_view(save_folder=str(cache_dir / "segment_graph.png"),
                                           communities_in=[atg.at_id_to_seg_ids_map[at_id] for at_id in topoenv.anything_ids])

            anything_objects: List[AnyThing] = []
            anything_colors: Dict[str, Tuple[int, int, int]] = {}

            for idx, anything_id in enumerate(topoenv.anything_ids):
                if anything_obj := topoenv.anything_manager.get(anything_id):
                    anything_objects.append(anything_obj)
                    # Assign a color from a uniform distribution
                    hue = idx / len(topoenv.anything_ids)
                    # Convert HSV to RGB (saturation and value are constant for vibrant colors)
                    # HSV values are in range [0, 1] for hue, saturation and value
                    # OpenCV expects H in [0, 179], S in [0, 255], V in [0, 255]
                    hsv_color = np.asarray([[[hue * 179, 255, 255]]]).astype(np.uint8)
                    rgb_color = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2RGB)
                    anything_colors[anything_id] = tuple(int(c) for c in rgb_color[0][0])  # type: ignore

            topoenv_frame_ids = topoenv.frame_ids
            anything_lifespan_frame_indices = []
            for topoenv_frame_id in topoenv_frame_ids:
                fd = topoenv.frame_manager.get(topoenv_frame_id, optional=False)
                assert fd is not None
                anything_lifespan_frame_indices.append(fd.frame_idx)

            anything_lifespan_frame_indices = list(set(anything_lifespan_frame_indices))

            plot_anything_lifespans(topoenv, REPRESENTATIVE_SEGMENTS_DIR, anything_colors, frame_indices=anything_lifespan_frame_indices)

            logger.info(f"Number of AnyThing objects found: {len(anything_objects)}.")

            logger.info("Calculating filled centers for all objects and frames...")
            topoManFilled, Fmap, F_sampson = topoenv.topoManFilled_w_log
            all_centers_pix = topoManFilled
            logger.info("Center calculation complete.")

            KEYPOINT_FRAMES_DIR = cache_dir / "keypoint_frames"
            KEYPOINT_FRAMES_DIR.mkdir(exist_ok=True)
            logger.info(
                f"Saving frames with masks, keypoints and bounding boxes to {KEYPOINT_FRAMES_DIR}."
            )

            SEGMENT_FRAMES_DIR = cache_dir / "segment_frames"
            SEGMENT_FRAMES_DIR.mkdir(exist_ok=True)
            logger.info(f"Saving frames with segments to {SEGMENT_FRAMES_DIR}")

            for frame_idx in tqdm(range(len(topoenv.videoContainer)), desc="Generating segmented video."):
                original_frame = topoenv.videoContainer[frame_idx]
                output_frame = original_frame.copy()

                frame_id = FrameDescriptorBase.create_id(topoenv.video_id, frame_idx)

                fd = topoenv.frame_manager.get(frame_id, optional=True)
                if fd is None:
                    continue
                if not fd.segments_computed:
                    continue

                keypoint_frame = np.zeros_like(original_frame)
                segment_frame = np.zeros_like(original_frame)
                drawn_on_segment_frame = False

                segments_drawn_fd_ids: List[str] = []

                # For the current frame, find all AnyThing objects that have segments in it.
                for anything_object in anything_objects:
                    # --- Draw Masks (only if visible in this frame) ---
                    if frame_id in anything_object.frame_ids:
                        # Get the color for this AnyThing
                        anything_color = anything_colors.get(
                            anything_object.id, (255, 255, 255)
                        )  # Default to white

                        # Create a combined boolean mask for the entire AnyThing in this frame.
                        full_thing_mask = anything_object.full_frame_mask(frame_id)

                        if full_thing_mask.size != 0:
                            output_frame[full_thing_mask] = anything_color

                            if frame_id not in segments_drawn_fd_ids and (fd := topoenv.frame_manager.get(frame_id)):
                                segments_drawn_fd_ids.append(frame_id)
                                for segment in fd.segments:
                                    full_frame_mask_ = segment.full_frame_mask
                                    if full_frame_mask_.size != 0:
                                        keypoint_frame[full_frame_mask_] = (50, 50, 50)
                                for kp in fd.keypoints:
                                    cv2.circle(
                                        keypoint_frame,
                                        (int(kp.loc[0]), int(kp.loc[1])),
                                        radius=2,
                                        color=(20, 20, 20),
                                        thickness=-1,
                                    )

                            # --- Draw Mask on keypoint_frame ---
                            keypoint_frame[full_thing_mask] = anything_color

                            # --- Draw Bounding Box on keypoint_frame ---
                            # Convert boolean mask to an 8-bit format for finding contours
                            mask_for_contours = full_thing_mask.astype(np.uint8)
                            contours, _ = cv2.findContours(
                                mask_for_contours,
                                cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE,
                            )
                            if contours:
                                # Combine all contours to get a single bounding box
                                all_points = np.concatenate(contours, axis=0)
                                x, y, w, h = cv2.boundingRect(all_points)
                                cv2.rectangle(
                                    keypoint_frame,
                                    (x, y),
                                    (x + w, y + h),
                                    anything_color,
                                    2,
                                )

                            # --- Draw Keypoints on keypoint_frame ---
                            segment_ids_for_frame = anything_object.frame_id_to_segment_id_map.get(
                                frame_id, []
                            )
                            for seg_id in segment_ids_for_frame:
                                if (segment := topoenv.segment_manager.get(seg_id)) and segment.keypoints:
                                    for kp in segment.keypoints:
                                        cv2.circle(
                                            keypoint_frame,
                                            (int(kp.loc[0]), int(kp.loc[1])),
                                            radius=2,
                                            color=(180, 180, 180),
                                            thickness=-1,
                                        )

                    # --- Draw Filled Center on keypoint_frame (always, for interpolation) ---
                    anything_idx = topoenv.anything_id_index(anything_object.id)
                    try:
                        topoenv_frame_idx = topoenv.frame_id_index(frame_id)
                        if anything_idx != -1:
                            center_coords = all_centers_pix[topoenv_frame_idx, :, anything_idx]
                            if not np.isnan(center_coords).any():
                                center_x, center_y = int(center_coords[0]), int(center_coords[1])
                                # Draw a filled white circle for the center
                                cv2.circle(
                                    keypoint_frame,
                                    (center_x, center_y),
                                    radius=5,
                                    color=(255, 255, 255),
                                    thickness=-1,
                                )
                                # Draw a black border for visibility
                                cv2.circle(
                                    keypoint_frame,
                                    (center_x, center_y),
                                    radius=5,
                                    color=(0, 0, 0),
                                    thickness=1,
                                )
                    except ValueError:
                        # frame_id not in topoenv internal mapping
                        pass

                save_path = KEYPOINT_FRAMES_DIR / f"frame_{frame_idx:04d}.png"
                if np.sum(keypoint_frame) > 0:
                    cv2.imwrite(str(save_path), cv2.cvtColor(keypoint_frame, cv2.COLOR_RGB2BGR))

                if fd := topoenv.frame_manager.get_by_index(frame_idx, topoenv.video_id):
                    if fd.segments_computed and fd.segment_ids:
                        for seg_id in fd.segment_ids:
                            if seg := topoenv.segment_manager.get(seg_id):
                                if seg.enough_detail_captured:
                                    full_frame_mask = seg.full_frame_mask

                                    if full_frame_mask.size != 0:
                                        if list(anything_colors.values()):
                                            segment_frame[full_frame_mask] = list(anything_colors.values())[0]
                                        else:
                                            segment_frame[full_frame_mask] = (255, 255, 255)
                                        drawn_on_segment_frame = True

                if drawn_on_segment_frame:
                    seg_save_path = SEGMENT_FRAMES_DIR / f"frame_{frame_idx:04d}.png"
                    cv2.imwrite(str(seg_save_path), cv2.cvtColor(segment_frame, cv2.COLOR_RGB2BGR))

                try:
                    stdin.write(output_frame.tobytes())
                except BrokenPipeError as exc:
                    raise RuntimeError(
                        f"FFmpeg stopped accepting frames. See {ffmpeg_log_path} for details."
                    ) from exc

        finally:
            try:
                stdin.close()
            except BrokenPipeError:
                pass

            return_code = process.wait()

        if return_code != 0:
            raise RuntimeError(
                f"FFmpeg encoding failed with exit code {return_code}. "
                f"See {ffmpeg_log_path} for details."
            )

        logger.info(f"Segmented video saved to {output_video_path}.")

    # Calculate and print the average span of frames for AnyThing objects
    if anything_objects:  # type: ignore
        total_span = 0
        for anything_obj in anything_objects:
            if anything_obj.segment_ids:
                # The span is the total number of frames the object appears in.
                span = len(anything_obj.frame_ids)
                total_span += span
        average_span = total_span / len(anything_objects)
        logger.info(f"Average span of frames for AnyThing objects: {average_span:.2f} frames")
    else:
        logger.info("No AnyThing objects found to calculate average span.")

    # Save any remaining dirty objects in caches before exiting
    logger.info("Saving all dirty objects in caches...")
    topoenv.frame_manager.evict_all()
    topoenv.segment_manager.evict_all()
    topoenv.matches_manager.evict_all()
    topoenv.match_manager.evict_all()
    topoenv.kp_manager.evict_all()
    topoenv.kps_manager.evict_all()
    topoenv.anything_manager.evict_all()
    logger.info("DONE.")
