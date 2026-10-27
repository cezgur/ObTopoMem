# Copyright 2026 Ata Baran Fırat
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union
import logging
import torch
from tqdm import tqdm
import numpy as np
import tensorflow as tf
import itertools
import time

from .VideoContainer import VideoContainer
from .FrameDescriptor import FrameDescriptor, FrameRoiDescriptor, FrameDescriptorCacheManager, create_segment_graph
from .FrameSegmentDescriptor import FrameSegmentDescriptorCacheManager, SegmentGraph
from .ImagePoint import KeyPointCacheManager, KeyPointsCacheManager
from .Matches import Matches, MatchCacheManager, MatchesCacheManager
from .AnyThing import AnyThing, AnyThingCacheManager, AnyThingGraph
from .wOmniGlue import wOmniGlue
from .wSAM import wSAM
from .TopoEnv import TopoEnv, TopoEnvCacheManager

logger = logging.getLogger(__name__)


class _VideoProcessorUtilities:
    """
    A private utility class for VideoProcessor, encapsulating calibration
    and data preparation logic.
    """
    @staticmethod
    def _get_max_consecutive_frames(frame_indices: List[int]) -> int:
        if not frame_indices:
            return 0

        sorted_indices = sorted(list(set(frame_indices)))
        if not sorted_indices:
            return 0

        max_consecutive = 0
        current_consecutive = 1

        for i in range(1, len(sorted_indices)):
            if sorted_indices[i] == sorted_indices[i-1] + 1:
                current_consecutive += 1
            else:
                max_consecutive = max(max_consecutive, current_consecutive)
                current_consecutive = 1

        max_consecutive = max(max_consecutive, current_consecutive)
        return max_consecutive

    @staticmethod
    def _calibrate_max_dim_for_processing(
        womniglue_instance: wOmniGlue,
        initial_max_dim: int = 630,
        min_max_dim: int = 512,
        step_size: int = 128
    ) -> int: # type: ignore
        """
        Determines the maximum image dimension that can be processed without VRAM errors.

        This function iteratively tests image sizes by generating random NumPy arrays
        and attempting to extract DINO and SuperPoint features.

        Args:
            womniglue_instance: An initialized wOmniGlue instance.
            initial_max_dim: The starting maximum dimension to test.
            min_max_dim: The minimum dimension to test before giving up.
            step_size: The amount to decrease the dimension by in each iteration.
        Returns:
            The largest tested dimension that succesfully processes the image.
            Raise RuntimeError if calibration fails at all tested dimensions.
        """
        if min_max_dim <= 0 or initial_max_dim < min_max_dim or step_size <= 0:
            raise ValueError(
                "Calibration requires 0 < min_max_dim <= initial_max_dim "
                "and step_size > 0."
            )

        PyTorchOOMError = torch.cuda.OutOfMemoryError if torch.cuda.is_available() else type("DummyCudaOOMError", (Exception,), {})

        current_max_dim = initial_max_dim
        logger.info("Starting VRAM calibration...")
        logger.info(f"Initial test dimension: {initial_max_dim}x{initial_max_dim}")

        while current_max_dim >= min_max_dim:
            logger.info(f"Attempting to process image of size: {current_max_dim}x{current_max_dim}")

            dummy_image = np.random.randint(
                0, 256, (current_max_dim, current_max_dim, 3), dtype=np.uint8
            )

            try:
                _ = womniglue_instance.getDINOFeatures(dummy_image)
                _ = womniglue_instance.getSPFeatures(dummy_image)

                logger.info(f"Successfully processed {current_max_dim}x{current_max_dim} image.")
                return current_max_dim

            except tf.errors.ResourceExhaustedError:
                logger.warning(f"VRAM exhausted at {current_max_dim}x{current_max_dim} (TensorFlow OOM).")
                logger.warning(f"Reducing dimension by {step_size} and retrying...")
            except PyTorchOOMError: # type: ignore
                logger.warning(f"VRAM exhausted at {current_max_dim}x{current_max_dim} (PyTorch OOM).")
                logger.warning(f"Reducing dimension by {step_size} and retrying...")
            except Exception as e:
                logger.error(f"An unexpected error occurred during calibration at {current_max_dim}x{current_max_dim}: {e}")
                logger.warning(f"Reducing dimension by {step_size} and retrying...")

            if current_max_dim == min_max_dim:
                raise RuntimeError(
                    f"VRAM calibration failed. Could not process an image "
                    f"at or above the minimum dimension of {min_max_dim}."
                )

            current_max_dim = max(min_max_dim, current_max_dim - step_size)


class VideoProcessor:
    def __init__(self,
                 video_path: str,
                 cache_dir: str,
                 *,
                 calibrate_vram: bool = True,
                 default_max_dim: int = 1280,
                 roi_bbox: Optional[Tuple[Union[int, float], Union[int, float], Union[int, float], Union[int, float]]] = None,
                 use_roi: bool = False,
                 roi_is_ratio: bool = False,
                 max_keypoints: Optional[int] = None,
                 keypoint_density_threshold: float = 0.001,
                #  use_cache_disk: bool = True, # Its negatory is not implemented. # TODO
                 sam_pred_iou_thresh: float = 0.95,
                 sam_stability_score_thresh: float = 0.7,
                 sam_box_nms_thresh: float = 0.7,
                 sam_crop_nms_thresh: float = 0.7):
        self.max_keypoints: Optional[int] = max_keypoints
        self.use_roi: bool = use_roi
        self.keypoint_density_threshold: float = keypoint_density_threshold

        self.videoContainer = VideoContainer(video_path)
        self.video_id = self.videoContainer.id

        self.womniglue = wOmniGlue()

        use_cache_disk = True # Not taken as input as the case that it is False is not handled # TODO

        # Pass SAM parameters to wSAM constructor
        self.wsam = wSAM(
            pred_iou_thresh=sam_pred_iou_thresh,
            stability_score_thresh=sam_stability_score_thresh,
            box_nms_thresh=sam_box_nms_thresh,
            crop_nms_thresh=sam_crop_nms_thresh
        )

        self.preprocessing_params: Optional[Tuple[Tuple[int, int], float, float]] = None

        # Initialize cache managers
        self.kp_manager = KeyPointCacheManager(cache_dir,
                                               use_cache_disk=use_cache_disk)
        self.kps_manager = KeyPointsCacheManager(cache_dir,
                                                 kp_manager=self.kp_manager,
                                                 use_cache_disk=use_cache_disk)
        self.match_manager = MatchCacheManager(cache_dir,
                                               keypoint_manager=self.kp_manager,
                                               use_cache_disk=use_cache_disk)
        self.matches_manager = MatchesCacheManager(cache_dir,
                                                   match_manager=self.match_manager,
                                                   use_cache_disk=use_cache_disk)
        self.segment_manager = FrameSegmentDescriptorCacheManager(cache_dir,
                                                                  self.videoContainer,
                                                                  kp_manager=self.kp_manager,
                                                                  use_cache_disk=use_cache_disk)
        self.anything_manager = AnyThingCacheManager(cache_dir,
                                                     self.segment_manager,
                                                     self.kps_manager,
                                                     self.matches_manager,
                                                     self.videoContainer,
                                                     use_cache_disk=use_cache_disk)
        self.frame_manager = FrameDescriptorCacheManager(cache_dir,
                                                         self.videoContainer,
                                                         self.kp_manager,
                                                         self.segment_manager,
                                                         self.womniglue,
                                                         self.wsam,
                                                         use_cache_disk=use_cache_disk)
        self.topoenv_manager = TopoEnvCacheManager(cache_dir,
                                                   use_cache_disk=use_cache_disk)

        self._original_roi_bbox = roi_bbox
        self._roi_is_ratio = roi_is_ratio
        self._preprocessed_roi_bbox: Optional[Tuple[int, int, int, int]] = None

        if calibrate_vram:
            self._max_dim_for_processing = _VideoProcessorUtilities._calibrate_max_dim_for_processing(
                initial_max_dim=default_max_dim,
                womniglue_instance=self.womniglue
            )
        else:
            self._max_dim_for_processing = default_max_dim

        if self.use_roi:
            if self._original_roi_bbox is None:
                raise ValueError("roi_bbox must be provided when use_roi is True.")

            if len(self._original_roi_bbox) != 4:
                raise ValueError("roi_bbox must contain exactly 4 values.")

            vid_w = self.videoContainer.width
            vid_h = self.videoContainer.height

            if self._roi_is_ratio:
                x_r, y_r, w_r, h_r = self._original_roi_bbox

                # Validate normalized ROI coordinates
                if not (
                    0 <= x_r < 1 and
                    0 <= y_r < 1 and
                    0 < w_r <= 1 and
                    0 < h_r <= 1 and
                    x_r + w_r <= 1 and
                    y_r + h_r <= 1
                ):
                    raise ValueError(
                        f"Invalid normalized ROI: {self._original_roi_bbox}."
                    )

                x = int(x_r * vid_w)
                y = int(y_r * vid_h)
                w = int(w_r * vid_w)
                h = int(h_r * vid_h)

            else:
                x, y, w, h = self._original_roi_bbox

                # Pixel coordinates must be integers
                if not all(float(v).is_integer() for v in (x, y, w, h)):
                    raise ValueError("Pixel ROI coordinates must be integers.")

                x, y, w, h = int(x), int(y), int(w), int(h)

            # Validate final pixel rectangle
            if not (
                0 <= x < vid_w and
                0 <= y < vid_h and
                w > 0 and
                h > 0 and
                x + w <= vid_w and
                y + h <= vid_h
            ):
                raise ValueError(
                    f"Invalid ROI ({x}, {y}, {w}, {h}) for video "
                    f"dimensions ({vid_w}, {vid_h})."
                )

            self._preprocessed_roi_bbox = (int(x), int(y), int(w), int(h))

        logger.info(f"Using max processing dimension: {self._max_dim_for_processing}.")
        if self.max_keypoints is not None:
            logger.info(f"Using max keypoints per frame: {self.max_keypoints}")

        self.anything_ids: List[str] = []
        self._cache_dir = cache_dir

        self.atg = None
        self.sg = None

    def match_frames(self, frame1_idx: int, frame2_idx: int) -> Tuple[Optional[Matches], bool]:
        assert self.videoContainer is not None
        if not (0 <= frame1_idx < len(self.videoContainer)):
            raise IndexError(f"frame1_idx ({frame1_idx}) is out of bounds for video of size {len(self.videoContainer)}.")
        if not (0 <= frame2_idx < len(self.videoContainer)):
            raise IndexError(f"frame2_idx ({frame2_idx}) is out of bounds for video of size {len(self.videoContainer)}.")

        if frame1_idx == frame2_idx:
            from_cache = False
            return None, from_cache

        if not (fd1 := self.frame_manager.get_by_index(frame1_idx, video_id=self.video_id)):
            raise ValueError(f"No frame descriptor at index {frame1_idx} is found in cache.")
        if not (fd2 := self.frame_manager.get_by_index(frame2_idx, video_id=self.video_id)):
            raise ValueError(f"No frame descriptor at index {frame2_idx} is found in cache.")

        fd1_id, fd2_id = fd1.id, fd2.id

        existing_matches = self.matches_manager.get_by_frame_pair(fd1_id, fd2_id)

        if existing_matches is None:
            matches_for_pair = fd1.match(fd2)
            return matches_for_pair, False

        return existing_matches, True

    def process_video(self, key_frame_interval_seconds: float = 1.0) -> TopoEnv:
        if not (num_frames := len(self.videoContainer)):
            raise ValueError("Cannot process an empty video.")

        key_frame_indices: List[int]
        intermediate_frame_indices: List[int]

        if key_frame_interval_seconds == -1:
            key_frame_indices = list(range(num_frames))
            intermediate_frame_indices = []
        else:
            every_n_frames = 5

            step = max(1, int(round(key_frame_interval_seconds * self.videoContainer.fps)))
            key_frame_indices = list(range(0, num_frames, step))
            key_frame_indices.append(num_frames - 1)
            key_frame_indices = sorted(set(key_frame_indices))

            intermediate_frame_indices = []

            # Ensure no gap between consecutive (key or intermediate) frames is > every_n_frames
            for i in range(1, len(key_frame_indices)):
                prev_k = key_frame_indices[i - 1]
                next_k = key_frame_indices[i]

                idx = prev_k + every_n_frames
                while idx < next_k:
                    intermediate_frame_indices.append(idx)
                    idx += every_n_frames

        key_frame_ids: List[str] = []
        intermediate_frame_ids: List[str] = []

        if len(key_frame_indices) <= 4:
            logger.info(f"Key frame indices: {key_frame_indices}.")
        else:
            logger.info(
                f"Following key frame indices are assigned: "
                f"[{key_frame_indices[0]}, {key_frame_indices[1]}, ...., "
                f"{key_frame_indices[-2]}, {key_frame_indices[-1]}]."
            )

        # ---- Phase 1 & 2
        #       (1) Feature extraction for all frames
        #       (2) Key frame segmentation (SAM automatic mask generation for all key frames)
        frame_indices_to_be_processed = [*key_frame_indices, *intermediate_frame_indices]

        logger.info(f"--- Phase 1 & 2: Feature Extraction and Key Frame Segmentation.")
        pbar_phase1 = tqdm(total=len(frame_indices_to_be_processed),
                           desc="Feature extraction for all frames", position=0)
        pbar_phase2 = tqdm(total=len(key_frame_indices),
                           desc="Key frame segmentation           ", position=1)
        for frame_idx in frame_indices_to_be_processed:
            fd = None
            # ---- Phase 1
            fd_possible = self.frame_manager.get_by_index(frame_idx, self.video_id)
            if self.use_roi:
                assert self._preprocessed_roi_bbox is not None
                if fd_possible is None or not isinstance(fd_possible, FrameRoiDescriptor):
                    fd = FrameRoiDescriptor(
                        videoContainer = self.videoContainer,
                        frame_idx = frame_idx,
                        kp_manager = self.kp_manager,
                        segment_manager = self.segment_manager,
                        womniglue = self.womniglue,
                        wsam = self.wsam,
                        max_dim_for_processing = self._max_dim_for_processing,
                        max_keypoints = self.max_keypoints,
                        keypoint_density_threshold = self.keypoint_density_threshold,
                        roi_bbox = self._preprocessed_roi_bbox
                    )
                    _ = fd.keypoints
                    _ = fd.DINODescriptorsRowVec
                else:
                    fd = fd_possible
            else:
                if fd_possible is None or isinstance(fd_possible, FrameRoiDescriptor):
                    fd = FrameDescriptor(
                        videoContainer = self.videoContainer,
                        frame_idx = frame_idx,
                        kp_manager = self.kp_manager,
                        segment_manager = self.segment_manager,
                        womniglue = self.womniglue,
                        wsam = self.wsam,
                        max_dim_for_processing = self._max_dim_for_processing,
                        max_keypoints = self.max_keypoints,
                        keypoint_density_threshold = self.keypoint_density_threshold
                    )

                    _ = fd.keypoint_ids
                    _ = fd.DINODescriptorsRowVec
                else:
                    fd = fd_possible

            if fd is not None:
                if self.preprocessing_params is None:
                    self.preprocessing_params = fd.preprocessing_params
                else:
                    assert self.preprocessing_params == fd.preprocessing_params

            if frame_idx in intermediate_frame_indices:
                intermediate_frame_ids.append(fd.id) # type: ignore

            pbar_phase1.update(1)

            # ---- Phase 2
            if frame_idx in key_frame_indices:
                _ = fd.segment_ids # type: ignore
                key_frame_ids.append(fd.id) # type: ignore

                pbar_phase2.update(1)

        assert self.preprocessing_params is not None

        pbar_phase1.close()
        pbar_phase2.close()

        # ---- Phase 3: Frame matching.
        logger.info(f"--- Phase 3: Key Frame <-> Key Frame and Key Frame <-> Intermediate Frame Matching.")
        key_frame_idx_pairs: List[Tuple[int, int]] = list(itertools.combinations(key_frame_indices, 2))

        # Key frame matching
        for fd1_idx, fd2_idx in tqdm(key_frame_idx_pairs, desc="Key frame matching."):
            _, _ = self.match_frames(fd1_idx, fd2_idx)

        # Intermediate frame matching
        key_frame_indices_np = np.array(key_frame_indices)
        for frame_idx in tqdm(intermediate_frame_indices, desc="Intermediate frame matching."):
            if frame_idx in key_frame_indices:
                continue
            later_kf_list_idx = np.searchsorted(key_frame_indices_np, frame_idx)
            earlier_kf_list_idx = later_kf_list_idx - 1

            if earlier_kf_list_idx < 0 or later_kf_list_idx >= len(key_frame_indices_np):
                continue

            earlier_kf_frame_num = key_frame_indices_np[earlier_kf_list_idx]

            later_kf_frame_num = key_frame_indices_np[later_kf_list_idx]

            earlier_matches, earlier_from_cache = self.match_frames(earlier_kf_frame_num, frame_idx)

            later_matches, later_from_cache = self.match_frames(frame_idx, later_kf_frame_num)

            if not earlier_from_cache or not later_from_cache:
                if earlier_matches is not None and later_matches is not None:
                    _ = earlier_matches.sum_with(later_matches)

        # Flush caches to disk before graph construction
        logger.info("Flushing all caches to disk before graph construction...")
        self.matches_manager.evict_all()
        self.match_manager.evict_all()
        self.segment_manager.evict_all()
        self.frame_manager.evict_all()
        self.kp_manager.evict_all()
        self.kps_manager.evict_all()

        # ---- Phase 4: Segment Graph creation and community detection.
        logger.info("--- Phase 4: Global Segment Graph Creation and Community Detection")
        t = time.perf_counter()
        self.sg = create_segment_graph(
            key_frame_ids,
            self.frame_manager,
            self.matches_manager,
            self.segment_manager
        )

        if self.sg is None:
            raise ValueError(
                "No usable segments were found in the video. "
                "Cannot construct the segment graph or TopoEnv."
            )

        segment_communities = self.sg.segment_communities

        print(f"Number of segment communities: {len(segment_communities)}")
        print(f"Number of community members in each community: "
              f"{sorted([len(community) for community in segment_communities], reverse=True)}")
        print(f"Time spent on Segment Graph: {time.perf_counter() - t}.")

        for community in segment_communities:
            at = AnyThing(segment_manager=self.segment_manager,
                          kps_manager=self.kps_manager,
                          matches_manager=self.matches_manager,
                          videoContainer=self.videoContainer,
                          initial_segment_ids=community)
            at.filter_wrt_noOfKeyPoints()
            if len(at.frame_ids) > 1:
                self.anything_ids.append(at.id)

        # Flush AnyThing cache before merging
        logger.info("Flushing AnyThing cache to disk before merging...")
        self.anything_manager.evict_all()

        # ---- Phase 5: Merge AnyThing objects using AnyThingGraph
        logger.info(f"--- Phase 5: AnyThing Graph Creation and Merging AnyThing objects.")
        if self.anything_ids:
            self.atg = AnyThingGraph(
                anything_ids=self.anything_ids,
                anything_manager=self.anything_manager,
                matches_manager=self.matches_manager,
                segment_manager=self.segment_manager,
                segment_graph=self.sg,
                cache_dir=self._cache_dir
            )

            anything_communities = self.atg.anything_communities

            final_anything_ids: List[str] = []

            for component in anything_communities:
                if not component:
                    continue

                if len(component) > 1:
                    all_segment_ids = set()
                    for anything_id in component:
                        if not (anything := self.anything_manager.get(anything_id, optional=False)):
                            continue
                        all_segment_ids.update(anything.segment_ids)

                    merged_anything = AnyThing(segment_manager=self.segment_manager,
                                               kps_manager=self.kps_manager,
                                               matches_manager=self.matches_manager,
                                               videoContainer=self.videoContainer,
                                               initial_segment_ids=list(all_segment_ids))
                    final_anything_ids.append(merged_anything.id)
                else:
                    # This AnyThing was not merged, add it directly.
                    final_anything_ids.append(component[0])

            logger.info(f"AnyThing count before merging: {len(self.anything_ids)}.")
            logger.info(f"AnyThing count after merging: {len(final_anything_ids)}.")

            self.anything_ids = final_anything_ids

            anything_ids_filtered = []
            for at_id in self.anything_ids:
                if at := self.anything_manager.get(at_id, optional=False):
                    if len(at.frame_ids) > 10:
                        anything_ids_filtered.append(at_id)

            self.anything_ids = anything_ids_filtered

            logger.info(f"AnyThing count after filtering: {len(self.anything_ids)}.")
        else:
            logger.info("No AnyThing objects to merge, skipping Phase 5.")

        for anything_id in self.anything_ids:
            at = self.anything_manager.get(anything_id, optional=False)
            assert at is not None
            at.filter_wrt_noOfKeyPoints()

        # ---- Phase 6: Prompted segmentation of in-between frames to propagate AnyThings.
        logger.info(f"--- Phase 6: Propagating AnyThings to in-between frames.")

        def create_foreground_prompts(matches: Matches, anything: AnyThing, key_frame_id: str) -> List[str]:
            if key_frame_id not in anything.frame_ids:
                return []

            if not (segment_ids_in_key_frame := anything.segment_id_in_frame(key_frame_id)):
                return []

            anything_kp_ids: List[str] = []
            for segment_id_in_key_frame in segment_ids_in_key_frame:
                if not (segment := self.segment_manager.get(segment_id_in_key_frame)):
                    continue
                anything_kp_ids.extend(segment.keypoint_ids)

            if not anything_kp_ids:
                return []

            prompts: List[str] = []
            for anything_kp_id in anything_kp_ids:
                if kp_mapped_id := matches.kp_id_map(anything_kp_id):
                    prompts.append(kp_mapped_id)

            return prompts

        for frame_idx in tqdm(intermediate_frame_indices, desc="Propagating AnyThings"):
            if frame_idx in key_frame_indices:
                continue

            if not (fd := self.frame_manager.get_by_index(frame_idx, self.video_id)):
                logger.warning(f"Could not find frame descriptor for frame {frame_idx}, "
                               "skipping propagation.")
                continue

            if fd.prompted_segments_computed:
                continue

            later_kf_list_idx = np.searchsorted(key_frame_indices_np, frame_idx)
            earlier_kf_list_idx = later_kf_list_idx - 1

            if earlier_kf_list_idx < 0 or later_kf_list_idx >= len(key_frame_indices_np):
                continue

            earlier_key_frame_num = key_frame_indices_np[earlier_kf_list_idx]
            later_key_frame_num = key_frame_indices_np[later_kf_list_idx]

            earlier_key_frame_id = FrameDescriptor.create_id(self.video_id, earlier_key_frame_num)
            later_key_frame_id = FrameDescriptor.create_id(self.video_id, later_key_frame_num)

            earlier_matches = self.matches_manager.get_by_frame_pair(fd.id,
                                                                     earlier_key_frame_id, optional=False)
            later_matches = self.matches_manager.get_by_frame_pair(fd.id,
                                                                   later_key_frame_id, optional=False)

            prompts_for_frame: Dict[str, Dict[str, List[str]]] = {}

            for anything_id in self.anything_ids:
                if not (anything := self.anything_manager.get(anything_id)):
                    continue

                all_fg_prompts = []
                if earlier_matches:
                    all_fg_prompts.extend(create_foreground_prompts(earlier_matches,
                                                                    anything,
                                                                    earlier_key_frame_id))

                if later_matches:
                    all_fg_prompts.extend(create_foreground_prompts(later_matches,
                                                                    anything,
                                                                    later_key_frame_id))

                if all_fg_prompts:
                    unique_fg_prompts = list(set(all_fg_prompts))
                    prompts_for_frame[anything.id] = {"foreground": unique_fg_prompts, "background": []}

            if prompts_for_frame:
                id_map = fd.compute_segments_from_prompts(prompts_for_frame)
                fd.filter_segments()
                if fd.segment_ids is None:
                    continue
                for anything_id, new_segment_id in id_map.items():
                    if new_segment_id in fd.segment_ids:
                        if anything := self.anything_manager.get(anything_id):
                            anything.add_segment(new_segment_id)

        for anything_id in self.anything_ids:
            at = self.anything_manager.get(anything_id, optional=False)
            assert at is not None
            at.filter_wrt_noOfKeyPoints()

        logger.info("Flushing modified caches after propagation...")
        self.segment_manager.evict_all()
        self.frame_manager.evict_all()
        self.anything_manager.evict_all()

        # ---- Phase 7: Creation of TopoEnv
        logger.info(f"--- Phase 7: Creating the Topological Environment Representation")

        assert self._max_dim_for_processing is not None

        topoenv = TopoEnv(cache_dir=self._cache_dir,
                          videoContainer=self.videoContainer,
                          anything_ids=self.anything_ids,
                          wsam = self.wsam,
                          preprocessing_params=self.preprocessing_params,
                          max_dim_for_processing=self._max_dim_for_processing)

        # ---- Phase 8: AnyThing Representative Segment Extraction
        logger.info(f"--- Phase 8: Extracting Representative Segments for AnyThings")
        for at_id in tqdm(self.anything_ids, desc="Going over each AnyThing"):
            at = self.anything_manager.get(at_id, optional=False)
            assert at is not None
            _ = at.representative_segment_ids

        self.anything_manager.evict_all()

        return topoenv
