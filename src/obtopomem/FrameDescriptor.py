# Copyright 2026 Ata Baran Fırat
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
import functools
from typing import Any, Dict, Optional, Tuple, List, Type, TypeVar, cast, Iterable, DefaultDict, FrozenSet, Union, overload

import cv2
import numpy as np
import pickle
import os
import itertools
from collections import defaultdict
import heapq

from .BaseCacheManager import BaseCacheManager
from .FrameDescriptorBase import FrameDescriptorBase, requires_womniglue
from .ImagePoint import KeyPoint, KeyPointCacheManager
from .Matches import Match, Matches, MatchCacheManager, MatchesCacheManager
from .wSAM import wSAM
from .wOmniGlue import wOmniGlue
from .FrameSegmentDescriptor import FrameSegmentDescriptor, FrameSegmentDescriptorCacheManager, SegmentGraph, TemporaryMatch
from .VideoContainer import VideoContainer

logger = logging.getLogger(__name__)

T_FrameDescriptor = TypeVar("T_FrameDescriptor", bound="FrameDescriptor")

def requires_wsam(func):
    """
    Decorator to ensure a method has access to a wSAM instance
    before proceeding.
    """
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        if self.wsam is None:
            raise ValueError(f"Method '{func.__name__}' requires a wSAM instance, but it is not available.")
        return func(self, *args, **kwargs)
    return wrapper

def create_segment_graph(frame_ids: List[str],
                         frame_manager: FrameDescriptorCacheManager,
                         matches_manager: MatchesCacheManager,
                         segment_manager: FrameSegmentDescriptorCacheManager,
                         sim_attr: Optional[str] = None
                         ) -> Optional[SegmentGraph]:
    """
    Constructs a SegmentGraph from segments belonging to selected frames.

    Retrieves the segment IDs associated with the provided frame IDs
    and constructs a graph using the specified cache managers.

    Args:
        frame_ids: IDs of frames whose segments should be included.
        frame_manager: Cache manager for retrieving frame descriptors.
        matches_mamager: Cache manager for feature-match collections.
        segment_manager: Cache manager for segment descriptors.
        sim_attr: Optional graph similarity-attribute name.

    Returns:
        Optional[SegmentGraph]: The constructed graph, or None if
            no segments are available.
    """
    segment_ids: List[str] = []
    for frame_id in frame_ids:
        fd = frame_manager.get(frame_id, optional=False)
        assert fd is not None
        if fd.segment_ids is None:
            continue
        segment_ids.extend(fd.segment_ids)

    if segment_ids:
        if sim_attr is None:
            return SegmentGraph(
                segment_ids=segment_ids,
                segment_manager=segment_manager,
                matches_manager=matches_manager
            )
        else:
            return SegmentGraph(
                segment_ids=segment_ids,
                segment_manager=segment_manager,
                matches_manager=matches_manager,
                sim_attr=sim_attr
            )
    else:
        return None

class FrameDescriptor(FrameDescriptorBase):
    """
    Represents a video frame and manages its visual features and segments.

    Provides frame preprocessing, SuperPoint feature extraction, SAM
    segmentation, and OmniGlue feature matching.

    Stores references to the extracted keypoints and segments through
    their corresponding cache managers.
    """
    def __init__(self,
                 videoContainer: VideoContainer,
                 frame_idx: int,
                 kp_manager: KeyPointCacheManager,
                 segment_manager: FrameSegmentDescriptorCacheManager,
                 womniglue: Optional[wOmniGlue] = None,
                 wsam: Optional[wSAM] = None,
                 max_dim_for_processing: Optional[int] = None,
                 max_keypoints: Optional[int] = None,
                 keypoint_density_threshold: float = 0.001,
                 is_dirty: bool = True,
                 **kwargs):
        super().__init__(video_container=videoContainer,
                         frame_idx=frame_idx,
                         womniglue=womniglue,
                         kp_manager=kp_manager,
                         max_dim_for_processing=max_dim_for_processing,
                         max_keypoints=max_keypoints,
                         is_dirty=is_dirty,
                         **kwargs
                         )
        self.cache_manager: cast(FrameDescriptorCacheManager, self.cache_manager) # type: ignore
        self.segment_manager = segment_manager

        self.wsam = wsam
        self._keypoint_density_threshold = keypoint_density_threshold
        self._segment_ids: Optional[Tuple[str, ...]] = None
        self._SPFeatures: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None

        # Computation Flags
        self.segments_computed = False
        self.automatic_segments_computed = False
        self.sp_features_computed = False
        self.keypoints_computed = False

    @property
    def prompted_segments_computed(self) -> bool:
        return self.segments_computed and not self.automatic_segments_computed

    @property
    def preprocessing_params(self) -> Tuple[Tuple[int, int], float, float]:
        h, w = self.videoContainer.height, self.videoContainer.width

        if max(h, w) <= self._max_dim_for_processing:
            proc_h, proc_w = h, w
        else:
            scale = self._max_dim_for_processing / max(h, w)
            proc_w, proc_h = int(w * scale), int(h * scale)

        scale_w = w / proc_w
        scale_h = h / proc_h
        return (proc_h, proc_w), scale_w, scale_h

    def _preprocess_frame(self) -> np.ndarray:
        (proc_h, proc_w), _, _ = self.preprocessing_params
        frame = self.videoContainer[self.frame_idx]
        orig_h, orig_w = frame.shape[:2]

        if orig_h == proc_h and orig_w == proc_w:
            return frame

        resized_frame = cv2.resize(frame, (proc_w, proc_h), interpolation=cv2.INTER_AREA)

        return resized_frame

    def _original_to_preprocessed_coords(self,
                                         kp_locs_orig: np.ndarray,
                                         proc_h: int,
                                         proc_w: int) -> Tuple[np.ndarray, np.ndarray]:
        if kp_locs_orig.size == 0:
            return np.array([], dtype=int), np.array([], dtype=int)

        (_, _), scale_w, scale_h = self.preprocessing_params

        # Convert to preprocessed continuous coords
        dothat = lambda ind, scale: kp_locs_orig[:, ind] / scale
        x_proc = dothat(0, scale_w)
        y_proc = dothat(1, scale_h)

        # Round to nearest pixel and clip to valid index
        dothat = lambda proc1, proc2: np.clip(np.rint(proc1).astype(int), 0, proc2 - 1)
        x_idx = dothat(x_proc, proc_w)
        y_idx = dothat(y_proc, proc_h)

        return x_idx, y_idx

    def _preprocessed_to_original_coords(self,
                                         kpts_processed: np.ndarray) -> np.ndarray:
        if kpts_processed.size == 0:
            return np.empty_like(kpts_processed, dtype=np.float32)

        (_, _), scale_w, scale_h = self.preprocessing_params
        H, W = self.videoContainer.height, self.videoContainer.width

        out = np.empty_like(kpts_processed, dtype=np.float32)

        # Scale each axis
        dothat = lambda ind, scale: kpts_processed[:, ind] * scale
        out[:, 0] = dothat(0, scale_w)
        out[:, 1] = dothat(1, scale_h)

        # Clip to original image bounds
        dothat = lambda ind, size: np.clip(out[:, ind], 0, size - 1, out=out[:, ind])
        dothat(0, W)
        dothat(1, H)
        return out

    def _original_to_preprocessed_indices(self,
                                          kp_locs_orig: np.ndarray,
                                          proc_h: int,
                                          proc_w: int) -> Tuple[np.ndarray, np.ndarray]:
        (_, _), scale_w, scale_h = self.preprocessing_params
        x_proc = kp_locs_orig[:, 0] / scale_w
        y_proc = kp_locs_orig[:, 1] / scale_h
        x_idx = np.clip(np.rint(x_proc).astype(int), 0, proc_w - 1)
        y_idx = np.clip(np.rint(y_proc).astype(int), 0, proc_h - 1)
        return x_idx, y_idx

    @property
    def segmentation_source_bbox(self) -> Tuple[int, int, int, int]:
        return (0, 0, self.videoContainer.width, self.videoContainer.height)

    @requires_wsam
    def compute_segments_from_prompts(self, prompts_dict: Dict[str, Dict[str, List[str]]]) -> Dict[str, str]:
        """
        Computes segments from a dictionary of prompts.
        Each key in the dictionary is an identifier (like an anything_id) and the value defines
        a segment to be found, with foreground and background keypoint IDs.

        Args:
            prompts_dict: A dictionary where keys are string identifiers and values are dicts
                          with keys 'foreground' and 'background' holding lists of KeyPoint IDs.

        Returns:
            A dictionary mapping the string identifiers from the input to the IDs of the newly created segments.
        """
        assert self.wsam is not None, "wSAM instance is not available"

        if self.automatic_segments_computed:
            self.logger.warning(f"Automatic segments have already been computed for this frame {self.id}. Overwriting.")

        (_, _), scale_w, scale_h = self.preprocessing_params
        x0, y0, _, _ = self.segmentation_source_bbox

        keypoint_id_map = {kp.id: kp for kp in self.keypoints}

        wsam_prompts_list = []
        ordered_anything_ids = []
        for anything_id, prompt_set in prompts_dict.items():
            fg_kp_ids = prompt_set.get("foreground", [])
            bg_kp_ids = prompt_set.get("background", [])

            fg_kps_objects = [keypoint_id_map[kp_id] for kp_id in fg_kp_ids if kp_id in keypoint_id_map]
            bg_kps_objects = [keypoint_id_map[kp_id] for kp_id in bg_kp_ids if kp_id in keypoint_id_map]

            if not fg_kps_objects and not bg_kps_objects:
                continue

            fg_points_orig = np.array([kp.loc for kp in fg_kps_objects]) if fg_kps_objects else np.empty((0, 2))
            bg_points_orig = np.array([kp.loc for kp in bg_kps_objects]) if bg_kps_objects else np.empty((0, 2))

            fg_points_preproc = np.empty_like(fg_points_orig)
            bg_points_preproc = np.empty_like(bg_points_orig)

            if fg_kps_objects:
                fg_points_preproc[:, 0] = (fg_points_orig[:, 0] - x0) / scale_w
                fg_points_preproc[:, 1] = (fg_points_orig[:, 1] - y0) / scale_h

            if bg_kps_objects:
                bg_points_preproc[:, 0] = (bg_points_orig[:, 0] - x0) / scale_w
                bg_points_preproc[:, 1] = (bg_points_orig[:, 1] - y0) / scale_h

            points = np.concatenate([fg_points_preproc, bg_points_preproc])
            labels = np.concatenate([np.ones(len(fg_points_preproc)), np.zeros(len(bg_points_preproc))])

            wsam_prompts_list.append({"points": points, "point_labels": labels})
            ordered_anything_ids.append(anything_id)

        if not wsam_prompts_list:
            self.segments_computed = False
            self.is_dirty = True
            return {}

        best_masks_data = self.wsam.segment_frame_for_multiple_prompts(
            self.preprocessed_frame,
            wsam_prompts_list
        )

        segment_descriptors: List[FrameSegmentDescriptor] = []
        id_map_to_return: Dict[str, str] = {}

        for i, best_mask_data in enumerate(best_masks_data):
            preprocessed_frame_sized_mask = best_mask_data["segmentation"]
            predicted_iou = best_mask_data["predicted_iou"]

            seg = FrameSegmentDescriptor(
                parent=self,
                parent_coord_mask=preprocessed_frame_sized_mask,
                predicted_iou=predicted_iou,
            )

            if seg.enough_detail_captured:
                segment_descriptors.append(seg)
                corresponding_anything_id = ordered_anything_ids[i]
                id_map_to_return[corresponding_anything_id] = seg.id

        self._segment_ids = tuple(segment.id for segment in segment_descriptors if segment.id is not None)
        self.segments_computed = True
        self.automatic_segments_computed = False
        self.is_dirty = True
        return id_map_to_return

    @requires_wsam
    def _compute_segments(self):
        assert self.wsam is not None

        segmentation_results = self.wsam.segment_frame(self.preprocessed_frame)
        segment_descriptors: List[FrameSegmentDescriptor] = []
        segment_ids: List[str] = []

        def ensure_uint8(array: np.ndarray) -> np.ndarray:
            if array.dtype in (np.float32, np.float64):
                array = (array > 0.5).astype(np.uint8)
            elif array.dtype == np.bool_:
                array = array.astype(np.uint8)
            elif array.dtype != np.uint8 or np.any((array != 0) & (array != 1)):
                array = (array > 0).astype(np.uint8)
            return array

        for result in segmentation_results:
            preprocessed_frame_sized_mask = result["segmentation"]
            if not np.any(preprocessed_frame_sized_mask):
                self.logger.debug(f"Skipping empty segment for frame {self.id}.")
                continue

            preprocessed_frame_sized_mask = ensure_uint8(preprocessed_frame_sized_mask)

            seg = FrameSegmentDescriptor(
                parent=self,
                parent_coord_mask=preprocessed_frame_sized_mask,
                predicted_iou=result["predicted_iou"]
            )

            # Filter segments based on detail captured (keypoint density) and size
            if seg.enough_detail_captured:
                segment_descriptors.append(seg)
                segment_ids.append(seg.id)
            else:
                self.logger.debug(f"Skipping segment for frame {self.id} due to insufficient detail.")

        self._segment_ids = tuple(segment_ids)
        self.segments_computed = True
        self.automatic_segments_computed = True
        self.is_dirty = True

    def filter_segments(self):
        """
        Filters frame segments using geometric and overlap constraints.

        Rejects segments containing holes, touching the frame boundary
        within five pixels, or occuying more than half the frame area.

        Processes segments in descending predicted IoU order, handles
        disconnected mask regions, and rejects segments whose centers
        fall outside their masks or whose overlap with already accepted
        regions exceeds the specified threshold.
        """
        if self.segment_ids is None:
            return

        # Filter out segments with holes
        good_segment_ids = []
        if self.segment_ids:
            for seg_id in self.segment_ids:
                if seg := self.segment_manager.get(seg_id, optional=False):

                    # --- Drop segments too close to frame boundary (within 5 pixels) ---
                    y_indices, x_indices = np.where(seg.full_frame_mask)

                    if y_indices.size == 0:
                        continue  # empty mask safeguard

                    min_x, max_x = x_indices.min(), x_indices.max()
                    min_y, max_y = y_indices.min(), y_indices.max()

                    if min_x < 5 or min_y < 5 or (self.videoContainer.width - 1 - max_x) < 5 or (self.videoContainer.height - 1 - max_y) < 5:
                        self.logger.debug(
                            f"Filtering: Discarding segment {seg.id} because it is within 5px of frame boundary."
                        )
                        continue

                    mask = seg.full_frame_mask.astype(np.uint8)
                    # Find the external contours of the segment.
                    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                    has_hole = False
                    if contours:
                        # Create a new mask by filling the external contours.
                        # This will fill any holes within the segment.
                        filled_mask = np.zeros_like(mask)
                        cv2.drawContours(filled_mask, contours, -1, (1,), cv2.FILLED)

                        # If the area of the filled mask is greater than the original mask's area,
                        # it means there were holes that have now been filled.
                        if np.sum(filled_mask) > np.sum(mask):
                            has_hole = True

                    if not has_hole:
                        good_segment_ids.append(seg_id)
                    else:
                        self.logger.debug(f"Filtering: Discarding segment {seg.id} because it has holes.")

        self._segment_ids = tuple(good_segment_ids)
        self.is_dirty = True

        if not self.segment_ids:
            return

        self.logger.debug(f"Filtering: Input segments count: {len(self.segment_ids)}")

        segment_ious: Dict[str, float] = {}

        for seg_id in self.segment_ids:
            if seg := self.segment_manager.get(seg_id, optional=False):
                segment_ious[seg_id] = seg.predicted_iou

        sorted_segment_ids = sorted(segment_ious.keys(), key=lambda k: segment_ious.get(k, 0.0), reverse=True)

        self.logger.debug(f"Filtering: Sorted segments count: {len(sorted_segment_ids)}")

        # Get original frame dimensions for area comparison
        H, W = self.videoContainer.height, self.videoContainer.width
        frame_area = H * W
        max_allowed_area = frame_area * 0.5

        # Define the NMS threshold: discard a segment if its overlap with taken area is above this.
        NMS_THRESHOLD = 0.1

        # Create a boolean mask to keep track of taken pixels in the original frame dimensions.
        taken = np.zeros((H, W), dtype=bool)

        filtered_segments: List[FrameSegmentDescriptor] = []

        for seg_id in sorted_segment_ids:
            if seg := self.segment_manager.get(seg_id, optional=False):
                # The area of the segment is in original frame coordinates.
                if seg.original_frame_area > max_allowed_area:
                    self.logger.debug(f"Filtering: Skipping segment {seg.id} because its area ({seg.original_frame_area}) is larger than 50% of the original frame area ({frame_area}).")
                    continue

                seg.isolate_disconnected_mask_islands(self)

                total_area = seg.original_frame_area
                if total_area == 0:
                    continue

                if not (center := seg.center_orig_coord):
                    continue
                x_coord, y_coord = center
                if not bool(seg.full_frame_mask[int(y_coord), int(x_coord)]):
                    continue

                intersection_area = np.sum(np.logical_and(seg.full_frame_mask, taken))

                # If the intersection area is a large fraction of the segment's own area, discard it.
                if (intersection_area / total_area) < NMS_THRESHOLD:
                    filtered_segments.append(seg)
                    # Add the entire segment's mask to the taken area.
                    taken |= seg.full_frame_mask
                    self.logger.debug(f"Filtering: Kept segment {seg.id}.")
                else:
                    self.logger.debug(f"Filtering: Discarding segment {seg.id} due to overlap > {NMS_THRESHOLD}.")

        self.logger.debug(f"Filtering: Final segments count after NMS: {len(filtered_segments)} for frame {self.id}.")

        self._segment_ids = tuple(seg.id for seg in filtered_segments)

        self.is_dirty = True

    @property
    def segment_ids(self) -> Optional[Tuple[str, ...]]:
        if not self.segments_computed:
            self._compute_segments()
            self.filter_segments()
        return self._segment_ids

    @property
    def segments(self) -> Iterable[FrameSegmentDescriptor]:
        if segment_ids := self.segment_ids:
            for segment_id in segment_ids:
                segment = self.segment_manager.get(segment_id, optional=False)
                if segment is not None and segment.enough_detail_captured:
                    yield segment

    @requires_womniglue
    def _compute_sp_features(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        assert self.womniglue is not None
        kpts_t, desc_t, scores_t = self.womniglue.getSPFeatures(
            self.preprocessed_frame,
            num_features=self._max_keypoints
        )

        return (np.array(kpts_t, copy=True),
                np.array(desc_t, copy=True),
                np.array(scores_t, copy=True))

    @property
    def SPFeatures(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Return (kpts, desc, scores) on the preprocessed frame, cached in RAM.
        kpts are in preprocessed-frame coords and are projected to original when building KeyPoint objects.
        """
        if not self.sp_features_computed:
            self._SPFeatures = self._compute_sp_features()
            self.sp_features_computed = True
            self.is_dirty = True
        assert self._SPFeatures is not None
        return self._SPFeatures

    @property
    def keypoint_ids(self) -> Tuple[str, ...]:
        """
        Build KeyPoint objects in original image coordinates from processed SPFeatures.
        """
        if not self.keypoints_computed:
            kpts_processed, sp_descriptors, scores = self.SPFeatures

            if kpts_processed.shape[0] == 0:
                self._keypoint_ids = tuple()
                self._keypoint_index_map = {}
                self.keypoints_computed = True # Set to True if no keypoints found
                self.is_dirty = True # Mark as dirty if state changes
            else:
                keypoint_ids: List[str] = []
                kpts_original = self._preprocessed_to_original_coords(kpts_processed)
                keypoint_objects: List[KeyPoint] = []
                keypoint_index_map = {}
                for i in range(kpts_original.shape[0]):
                    kp = KeyPoint(
                        loc=(kpts_original[i, 0], kpts_original[i, 1]),
                        sp_descriptor=sp_descriptors[i],
                        score=scores[i],
                        frame_id=self.id,
                        is_dirty=True
                    )
                    keypoint_objects.append(kp)
                    keypoint_ids.append(kp.id)
                    keypoint_index_map[kp.id] = i
                self._keypoint_ids = tuple(keypoint_ids)
                self._keypoint_index_map = keypoint_index_map
                self.keypoints_computed = True # Set to True if keypoints found
                self.is_dirty = True # Mark as dirty if state changes
        assert self._keypoint_ids is not None
        return self._keypoint_ids

    @property
    def keypoints(self) -> Iterable[KeyPoint]:
        for kp_id in self.keypoint_ids:
            assert self.kp_manager is not None
            kp = self.kp_manager.get(kp_id, optional=False)
            if kp is not None:
                yield kp

    @overload
    def match(self, other_descriptor: FrameDescriptor, matches_from_cache: Optional[Matches] = None) -> Matches: ...
    @overload
    def match(self, other_descriptor: FrameSegmentDescriptor, matches_from_cache: Optional[Matches] = None) -> List[TemporaryMatch]: ...
    @requires_womniglue
    def match(self,
              other_descriptor: Union[FrameDescriptor, FrameSegmentDescriptor],
              matches_from_cache: Optional[Matches] = None) -> Union[Matches, List[TemporaryMatch]]:
        if isinstance(other_descriptor, FrameDescriptor):
            if matches_from_cache is not None and isinstance(matches_from_cache, Matches):
                if matches_from_cache.frame_ids == frozenset((self.id, other_descriptor.id)):
                    return matches_from_cache

            matches = self.match_w_frame(other_descriptor)
        elif isinstance(other_descriptor, FrameSegmentDescriptor):
            if matches_from_cache is not None and isinstance(matches_from_cache, Matches):
                matches = self.match_w_segment(other_descriptor, matches_from_cache=matches_from_cache)
            else:
                matches = self.match_w_segment(other_descriptor)
        else:
            raise ValueError(f"FrameDescriptor object cannot be matched to {type(other_descriptor)}")

        return matches

    @requires_womniglue
    def match_w_segment(self,
                        segment: FrameSegmentDescriptor,
                        matches_from_cache: Optional[Matches] = None) -> List[TemporaryMatch]:
        """
        Retrieves feature matches between this frame and a specified segment.

        Uses cached matches between the two parent frames when available.
        Otherwise, computes frame-to-frame matches and retains only
        correspondences involving keypoints belonging to the segment.

        Args:
            segment: Segment whose keypoint correspondences are requested.
            matches_from_cahce: Optional existing frame-to-frame matches.

        Returns:
            List[TemporaryMatch]: Correspondences involving the segment's
                keypoints, with their associated matching confidences.
        """
        segment_kp_ids: Tuple[str, ...] = segment.keypoint_ids
        segment_fd_id: str = segment.parent_frame_id

        matches_full = None

        if matches_from_cache is not None and isinstance(matches_from_cache, Matches):
            if matches_from_cache.frame_ids == frozenset((segment_fd_id, self.id)):
                matches_full = matches_from_cache

        if matches_full is None:
            segment_fd: FrameDescriptor = self.cache_manager.get(segment_fd_id, optional=False)
            matches_full = self.match_w_frame(segment_fd)

        tempMatches = []

        for segment_kp_id in segment_kp_ids:
            if not (kp_matched := matches_full.kp_id_map(segment_kp_id)):
                continue
            match = matches_full.kp_id_to_match_map(segment_kp_id)
            assert match is not None
            conf = match.confidence
            tempMatches.append(TemporaryMatch(temp_kp_indices_or_kp_ids=frozenset((segment_kp_id, kp_matched)),
                                              confidence=conf))

        return tempMatches

    @requires_womniglue
    def match_w_frame(self, other_frame_descriptor: FrameDescriptor) -> Matches:
        """
        Matches this frame against another frame using OmniGlue.

        Uses SuperPoint features and associated DINO descriptors.
        If valid segments have already been computed, matching is
        restricted to keypoints belonging to those segments.

        Otherwise, all available keypoints with DINO descriptors are considered.

        Args:
            other_frame_descriptor: Frame match against

        Returns:
            Matches: Feature correspondences between the two frames,
                sorted by decreasing matching confidence.
        """
        assert self.womniglue is not None

        _ = self.DINODescriptorsRowVec
        _ = other_frame_descriptor.DINODescriptorsRowVec

        # --- Frame 0 Feature Preparation ---
        # Ensure keypoints and the index map are computed
        _ = self.keypoints
        sp_kpts_all0, sp_desc_all0, sp_scores_all0 = self.SPFeatures

        if self.segments_computed and self.segment_ids:
            self.logger.debug(f"Frame {self.frame_idx}: Using keypoints from {len(self.segment_ids)} segments for matching.")
            kps_set0 = {kp for seg in self.segments for kp in seg.keypoints if kp.dino_descriptor is not None}
            keypoints_to_use0 = list(kps_set0)

            # Get indices from the pre-computed map
            indices0 = [self._keypoint_index_map[kp.id] for kp in keypoints_to_use0 if kp.id in self._keypoint_index_map]

            # Filter the features using the found indices
            kpts_t0 = sp_kpts_all0[indices0]
            desc_t0 = sp_desc_all0[indices0]
            scores_t0 = sp_scores_all0[indices0]

            sp_features0 = (kpts_t0, desc_t0, scores_t0)
            dino_descriptors0 = np.array([kp.dino_descriptor for kp in keypoints_to_use0])
        else:
            self.logger.debug(f"Frame {self.frame_idx}: No segments found, using all keypoints for matching.")
            keypoints_to_use0 = [kp for kp in self.keypoints if kp.dino_descriptor is not None]

            # Get indices from the pre-computed map
            indices0 = [self._keypoint_index_map[kp.id] for kp in keypoints_to_use0 if kp.id in self._keypoint_index_map]

            # Filter the features using the found indices
            kpts_t0 = sp_kpts_all0[indices0]
            desc_t0 = sp_desc_all0[indices0]
            scores_t0 = sp_scores_all0[indices0]

            sp_features0 = (kpts_t0, desc_t0, scores_t0)
            dino_descriptors0 = np.array([kp.dino_descriptor for kp in keypoints_to_use0])

        (h0, w0), _, _ = self.preprocessing_params

        # --- Frame 1 Feature Preparation ---
        _ = other_frame_descriptor.keypoints
        sp_kpts_all1, sp_desc_all1, sp_scores_all1 = other_frame_descriptor.SPFeatures

        if other_frame_descriptor.segments_computed and other_frame_descriptor.segment_ids:
            self.logger.debug(f"Frame {other_frame_descriptor.frame_idx}: Using keypoints from {len(other_frame_descriptor.segment_ids)} segments for matching.")
            kps_set1 = {kp for seg in other_frame_descriptor.segments for kp in seg.keypoints if kp.dino_descriptor is not None}
            keypoints_to_use1 = list(kps_set1)

            indices1 = [other_frame_descriptor._keypoint_index_map[kp.id] for kp in keypoints_to_use1 if kp.id in other_frame_descriptor._keypoint_index_map]

            kpts_t1 = sp_kpts_all1[indices1]
            desc_t1 = sp_desc_all1[indices1]
            scores_t1 = sp_scores_all1[indices1]

            sp_features1 = (kpts_t1, desc_t1, scores_t1)
            dino_descriptors1 = np.array([kp.dino_descriptor for kp in keypoints_to_use1])
        else:
            self.logger.debug(f"Frame {other_frame_descriptor.frame_idx}: No segments found, using all keypoints for matching.")
            keypoints_to_use1 = [kp for kp in other_frame_descriptor.keypoints if kp.dino_descriptor is not None]

            indices1 = [other_frame_descriptor._keypoint_index_map[kp.id] for kp in keypoints_to_use1 if kp.id in other_frame_descriptor._keypoint_index_map]

            kpts_t1 = sp_kpts_all1[indices1]
            desc_t1 = sp_desc_all1[indices1]
            scores_t1 = sp_scores_all1[indices1]

            sp_features1 = (kpts_t1, desc_t1, scores_t1)
            dino_descriptors1 = np.array([kp.dino_descriptor for kp in keypoints_to_use1])

        (h1, w1), _, _ = other_frame_descriptor.preprocessing_params

        match_indices, match_confidences = self.womniglue.findMatches(
            sp_features0=sp_features0,
            dino_descriptors0=dino_descriptors0,
            height0=h0, width0=w0,
            sp_features1=sp_features1,
            dino_descriptors1=dino_descriptors1,
            height1=h1, width1=w1
        )

        assert self.kp_manager is not None
        found_matches: List[Match] = [
            Match(keypoint_ids=(keypoints_to_use0[idx0].id,
                                keypoints_to_use1[idx1].id),
                  kp_manager=self.kp_manager,
                  confidence=match_confidences[i])
            for i, (idx0, idx1) in enumerate(match_indices)
        ]

        found_matches = sorted(found_matches,
                               key= lambda x: x.confidence,
                               reverse=True)

        found_match_ids = [match.id for match in found_matches]

        return Matches(found_match_ids,
                       frame_ids=(self.id, other_frame_descriptor.id),
                       match_manager=cast(MatchCacheManager, Match.cache_manager))

    def to_dict(self) -> Dict[str, Any]:
        data = super().to_dict()
        data.update({
            "__class__": "FrameDescriptor",
            "segment_ids": self._segment_ids if self._segment_ids is not None else [],
            "_keypoint_density_threshold": self._keypoint_density_threshold,
            "_sp_features": self._SPFeatures,
            "segments_computed": self.segments_computed,
            "automatic_segments_computed": self.automatic_segments_computed,
            "keypoints_computed": self.keypoints_computed,
            "sp_features_computed": self.sp_features_computed,
            "_keypoint_index_map":  self._keypoint_index_map if hasattr(self, "_keypoint_index_map") else {}
        })
        return data

    @classmethod
    def _from_dict_end(cls: Type[T_FrameDescriptor],
                       instance: T_FrameDescriptor,
                       data: Dict[str, Any]) -> T_FrameDescriptor:
        instance.is_dirty = False
        instance.segments_computed = data.get("segments_computed", False)
        instance.automatic_segments_computed = data.get("automatic_segments_computed")
        if instance.segments_computed:
            instance._segment_ids = data.get("segment_ids")

            # If segments were supposed to be computed but none were loaded, reset the flag
            if instance._segment_ids is None:
                instance.segments_computed = False
                instance.is_dirty = True # Mark as dirty so it gets re-saved with updated flag
                cls.logger.warning(f"FrameDescriptor {instance.id}: Segments were marked as computed, but no segment id was found.")

        instance.sp_features_computed = data.get("sp_features_computed", False)
        if instance.sp_features_computed:
            if (spf := data.get("_sp_features")) is not None:
                if isinstance(spf, (list, tuple)) and len(spf) == 3:
                    instance._SPFeatures = (np.array(spf[0], copy=True),
                                            np.array(spf[1], copy=True),
                                            np.array(spf[2], copy=True))
                else:
                    cls.logger.warning(f"Unexpected format for _sp_features: {type(spf)}, expected tuple/list of 3 elements.")
                    instance._SPFeatures = None
            # If SP features were supposed to be computed but none were loaded, reset the flag
            if instance._SPFeatures is None:
                instance.sp_features_computed = False
                instance.is_dirty = True
                cls.logger.warning(f"FrameDescriptor {instance.id}: SP features were marked as computed, but none could be loaded. Resetting sp_features_computed to False to force recomputation.")

        instance.keypoints_computed = data.get("keypoints_computed", False)
        instance._keypoint_index_map = data.get("_keypoint_index_map", {})

        if instance.keypoints_computed:
            if instance._keypoint_ids is None:
                instance.keypoints_computed = False
                instance.is_dirty = True
                cls.logger.warning(
                    f"FrameDescriptor {instance.id}: KeyPoints were marked as computed, but their IDs are missing."
                )
            elif not instance._keypoint_index_map and instance._keypoint_ids:
                instance._keypoint_index_map = {
                    kp_id: i for i, kp_id in enumerate(instance._keypoint_ids)
                }
                instance.is_dirty = True

        return instance

    @classmethod
    def from_dict(cls: Type[T_FrameDescriptor],
                  data: Dict[str, Any],
                  videoContainer: VideoContainer,
                  womniglue: Optional[wOmniGlue],
                  wsam: Optional[wSAM],
                  keypoint_manager: KeyPointCacheManager,
                  segment_manager: FrameSegmentDescriptorCacheManager) -> T_FrameDescriptor:
        if videoContainer.id != data["video_id"]:
            raise ValueError("The VideoContainer object is not the one that was used for the FrameDescriptor.")
        instance = cls(
            videoContainer=videoContainer,
            frame_idx=data["frame_idx"],
            kp_manager=keypoint_manager,
            segment_manager=segment_manager,
            womniglue=womniglue,
            wsam=wsam,
            id=data.get("id"),
            max_dim_for_processing=data.get("cache_max_dim_for_processing"),
            max_keypoints=data.get("cache_max_keypoints"),
            keypoint_density_threshold=data.get("_keypoint_density_threshold", 0.001),
            is_dirty=False
        )
        cls._rehydrate_caches(instance,
                              data,
                              keypoint_manager)

        return cls._from_dict_end(instance, data)


class FrameRoiDescriptor(FrameDescriptor):
    """
    Represents a rectangular region of interest within a video frame.

    Extends FrameDescriptor by restricting preprocessing to the specified
    region while maintaining keypoint coordinates in the original frame
    coordinate system.
    """
    def __init__(self,
                 *args,
                 roi_bbox: Tuple[int, int, int, int],
                 is_dirty: bool = True,
                 **kwargs):
        super().__init__(*args, is_dirty=is_dirty, **kwargs)
        if roi_bbox is None:
            raise ValueError("FrameRoiDescriptor requires a non-None roi_bbox")
        self._roi_bbox = roi_bbox

    @property
    def segmentation_source_bbox(self) -> Tuple[int, int, int, int]:
        return self._roi_bbox

    @property
    def preprocessing_params(self) -> Tuple[Tuple[int, int], float, float]:
        _, _, w, h = self._roi_bbox
        if max(h, w) <= self._max_dim_for_processing:
            preproc_h, preproc_w = h, w
        else:
            scale = self._max_dim_for_processing / max(h, w)
            preproc_w, preproc_h = int(w * scale), int(h * scale)
        scale_w = (w / preproc_w) if preproc_w > 0  else 1.0
        scale_h = (h / preproc_h) if preproc_h > 0  else 1.0
        return (preproc_h, preproc_w), scale_w, scale_h

    def _preprocess_frame(self) -> np.ndarray:
        x, y, w, h = self._roi_bbox
        frame = self.videoContainer[self.frame_idx]
        roi = frame[y:y+h, x:x+w]
        (preproc_h, preproc_w), _, _ = self.preprocessing_params
        if roi.shape[0] == preproc_h and roi.shape[1] == preproc_w:
            return roi
        return cv2.resize(roi, (preproc_w, preproc_h), interpolation=cv2.INTER_AREA)

    def _preprocessed_to_original_coords(self, kpts_processed: np.ndarray) -> np.ndarray:
        (_, _), scale_w, scale_h = self.preprocessing_params
        x0, y0, _, _ = self._roi_bbox
        out = np.empty_like(kpts_processed, dtype=np.float32)
        out[:, 0] = kpts_processed[:, 0] * scale_w + x0
        out[:, 1] = kpts_processed[:, 1] * scale_h + y0
        np.clip(out[:, 0], 0, self.videoContainer.width - 1, out=out[:, 0])
        np.clip(out[:, 1], 0, self.videoContainer.height - 1, out=out[:, 1])
        return out

    def _original_to_preprocessed_indices(self,
                                          kp_locs_orig: np.ndarray,
                                          proc_h: int,
                                          proc_w: int) -> Tuple[np.ndarray, np.ndarray]:
        (_, _), scale_w, scale_h = self.preprocessing_params
        x0, y0, _, _ = self._roi_bbox
        x_proc = (kp_locs_orig[:, 0] - x0) / scale_w
        y_proc = (kp_locs_orig[:, 1] - y0) / scale_h
        x_idx = np.clip(np.rint(x_proc).astype(int), 0, proc_w - 1)
        y_idx = np.clip(np.rint(y_proc).astype(int), 0, proc_h - 1)
        return x_idx, y_idx

    def to_dict(self) -> Dict[str, Any]:
        data = super().to_dict()
        data.update({
            "__class__": "FrameRoiDescriptor",
            "_roi_bbox": self._roi_bbox
        })
        return data

    @classmethod
    def from_dict(cls: Type["FrameRoiDescriptor"],
                  data: Dict[str, Any],
                  videoContainer: VideoContainer,
                  womniglue: Optional[wOmniGlue],
                  wsam: Optional[wSAM],
                  keypoint_manager: KeyPointCacheManager,
                  segment_manager: FrameSegmentDescriptorCacheManager) -> "FrameRoiDescriptor":

        if videoContainer.id != data["video_id"]:
            raise ValueError("The VideoContainer object is not the one that was used for the FrameRoiDescriptor.")

        roi_bbox = data.get("_roi_bbox")
        if roi_bbox is None:
            raise ValueError("Cannot deserialize FrameRoiDescriptor: '_roi_bbox' is missing.")

        instance = cls(
            videoContainer=videoContainer,
            frame_idx=data["frame_idx"],
            kp_manager=keypoint_manager,
            segment_manager=segment_manager,
            womniglue=womniglue,
            wsam=wsam,
            id=data.get("id"),
            max_dim_for_processing=data.get("cache_max_dim_for_processing"),
            max_keypoints=data.get("cache_max_keypoints"),
            keypoint_density_threshold=data.get("_keypoint_density_threshold", 0.001),
            roi_bbox=tuple(roi_bbox),
            is_dirty=False
        )
        cls._rehydrate_caches(instance, data, keypoint_manager)
        return cls._from_dict_end(instance, data)

class FrameDescriptorCacheManager(BaseCacheManager[FrameDescriptor]):
    """
    Manages cached FrameDescriptor and FrameRoiDescriptor instances.

    Supports retrieval by frame ID or video-frame index and restores
    the appropriate descriptor subclass from serialized data.
    """
    def __init__(self,
                 cache_dir: str,
                 videoContainer: VideoContainer,
                 keypoint_manager: KeyPointCacheManager,
                 segment_manager: FrameSegmentDescriptorCacheManager,
                 womniglue: Optional[wOmniGlue] = None,
                 wsam: Optional[wSAM] = None,
                 use_cache_disk: bool = False,
                 max_size: Optional[int] = 50):
        super().__init__(cache_dir, FrameDescriptor, use_cache_disk=use_cache_disk, max_size=max_size) # Passed max_size
        self.videoContainer = videoContainer
        self.keypoint_manager = keypoint_manager
        self.segment_manager = segment_manager
        self.womniglue = womniglue
        self.wsam = wsam
        self.video_id: Optional[str] = None

    def _get_filepath(self, obj_id: str) -> str:
        return os.path.join(self.cache_dir, f"{obj_id}.pkl")

    def get_by_index(self, index: int, video_id: Optional[str], optional=True) -> Optional[FrameDescriptor]:
        if video_id is None:
            if self.video_id is None:
                raise ValueError("There is no saved video_id in cache manager, it has to be supplied when calling.")
            else:
                video_id = self.video_id
        else:
            self.video_id = video_id

        frame_id = FrameDescriptorBase.create_id(video_id, index)
        if frame_id is None:
            return None
        return self.get(frame_id, optional=optional)

    def _load_from_file(self, filepath: str) -> FrameDescriptor:
        with open(filepath, "rb") as f:
            data = pickle.load(f)

        class_name = data.get("__class__")

        if class_name == "FrameDescriptor":
            return FrameDescriptor.from_dict(data, self.videoContainer, self.womniglue, self.wsam, self.keypoint_manager, self.segment_manager)
        elif class_name == "FrameRoiDescriptor":
            return FrameRoiDescriptor.from_dict(data, self.videoContainer, self.womniglue, self.wsam, self.keypoint_manager, self.segment_manager)
        else:
            raise TypeError(f"Unknown descriptor class for deserialization: {class_name}")
