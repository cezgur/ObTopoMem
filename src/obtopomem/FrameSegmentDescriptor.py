# Copyright 2026 Ata Baran Fırat
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import pickle
from typing import Any, Dict, Optional, Tuple, TypeVar, List, Set, DefaultDict, FrozenSet, Union, TYPE_CHECKING
if TYPE_CHECKING:
    from .FrameDescriptor import FrameDescriptor

import uuid
import numpy as np
import cv2
import functools
import itertools
from collections import defaultdict, namedtuple
import logging
import matplotlib.pyplot as plt

from .BaseCacheManager import BaseCacheManager
from .FrameDescriptorBase import FrameDescriptorBase
from .ImagePoint import KeyPoint, KeyPointCacheManager
from .VideoContainer import VideoContainer
from .wOmniGlue import wOmniGlue
from .Graph import Graph
from .Matches import MatchesCacheManager, Matches

logger = logging.getLogger(__name__)

T_Seg = TypeVar("T_Seg", bound="FrameSegmentDescriptor")

CONFIGURATION = 3   # 1: log confirmation + basic area sim
                    # 2: linear confirmation + fancy area sim
                    # 3: linear confirmation / avg #keypoints
                    # 4: (linear confirmation / avg #keypoints)^2

TemporaryMatch = namedtuple("TemporaryMatches", ["confidence", "temp_kp_indices_or_kp_ids"])
TemporaryKeyPoint = namedtuple("TemporaryKeyPoint", ["idx", "loc"])


class SegmentGraph(Graph):
    def __init__(self,
                 segment_ids: List[str],
                 segment_manager: FrameSegmentDescriptorCacheManager,
                 matches_manager: MatchesCacheManager,
                 sim_attr: str = "weight"):
        self.segment_manager: FrameSegmentDescriptorCacheManager = segment_manager
        self.matches_manager: MatchesCacheManager = matches_manager

        super().__init__(sim_attr=sim_attr,
                         directed=False,
                         segment_ids=segment_ids)

    @property
    def segment_communities(self) -> List[List[str]]:
        """
        Runs InfoMap community detection and then iteratively refines communities by splitting
        any community that contains non-intersecting segments from the same frame.
        Returns a list of segment_id groups.
        """
        intersection_cache: Dict[FrozenSet[str], bool] = {}
        # ---- 1. Initial clustering
        logger.info("Initial clustering...")
        # initial_clustering_result = self._cluster_leiden(resolution=0.1)
        initial_clustering_result = self._cluster_infomap(trials=100)

        final_communities: List[List[str]] = []
        for community_vids in initial_clustering_result["communities"]:
            segment_ids = [self.node_id_node_idx(v) for v in community_vids]
            assert None not in segment_ids, f"A None value is encountered in segment_ids."
            final_communities.append(segment_ids)

        # ---- 2. Small Community Elimination
        # num_communities_wo_elimination = len(final_communities)
        # logger.info("Small Community Elimination...")
        # community_by_frame: List[Set[str]] = []
        # for community in final_communities:
        #     element_by_frame: List[str] = []
        #     for element in community:
        #         element_by_frame.append(self.segment_id_to_frame_id_map[element])
        #     community_by_frame.append(set(element_by_frame))

        # eliminated_final_communities: List[List[str]] = []
        # for i, community in enumerate(final_communities):
        #     if len(community_by_frame[i]) > 5:
        #         eliminated_final_communities.append(community)

        # final_communities = eliminated_final_communities
        # logger.info(f"      Eliminated {num_communities_wo_elimination - len(final_communities)} communities with small community elimination.")
        # logger.info(f"      {len(final_communities)} communities go on.")

        # ---- 3. Iterative Splitting Loop
        logger.info("Iterative splitting loop...")
        similarity_cache: Dict[FrozenSet[str], float] = {}
        division_counter = 0
        while True:
            a_split_was_made = False

            communities_after_split: List[List[str]] = []

            for community in final_communities:
                if not community:
                    continue

                # ---- Find Violations by checking for disconnected components within a frame ----
                violation_found_in_community = False
                segs_by_frame: DefaultDict[str, List[str]] = defaultdict(list)
                for seg_id in community:
                    if not (segment := self.segment_manager.get(seg_id, optional=False)):
                        continue
                    segs_by_frame[segment.parent_frame_id].append(seg_id)

                for frame_id, seg_ids_in_frame in segs_by_frame.items():
                    if len(seg_ids_in_frame) < 2:
                        continue

                    # Build touching graph for the segments in the current frame
                    adj = defaultdict(list)
                    for i, s1_id in enumerate(seg_ids_in_frame):
                        for s2_id in seg_ids_in_frame[i+1:]:
                            s_pair = frozenset({s1_id, s2_id})
                            if s_pair in intersection_cache:
                                intersects = intersection_cache[s_pair]
                            else:
                                s1 = self.segment_manager.get(s1_id, optional=False)
                                s2 = self.segment_manager.get(s2_id, optional=False)
                                if not s1 or not s2: continue
                                intersects = s1.intersects(s2)
                                intersection_cache[s_pair] = intersects

                            if intersects:
                                adj[s1_id].append(s2_id)
                                adj[s2_id].append(s1_id)

                    # Find connected components to form the base of new communities.
                    base_groups: List[List[str]] = []
                    visited = set()
                    for seg_id in seg_ids_in_frame:
                        if seg_id not in visited:
                            component = []
                            q = [seg_id]
                            visited.add(seg_id)
                            while q:
                                curr = q.pop(0)
                                component.append(curr)
                                for neighbor in adj.get(curr, []):
                                    if neighbor not in visited:
                                        visited.add(neighbor)
                                        q.append(neighbor)
                            base_groups.append(component)

                    # A violation exists if there's more than one disconnected component
                    if len(base_groups) > 1:
                        # --- VIOLATION FOUND: Perform the split ----
                        a_split_was_made = True
                        violation_found_in_community = True
                        division_counter += 1

                        # These are the segments that will form the core of the new communities
                        base_segments = [group[0] for group in base_groups]
                        num_new_communities = len(base_segments)

                        newly_formed_communities: List[List[str]] = [[] for _ in range(num_new_communities)]
                        for i, group in enumerate(base_groups):
                            newly_formed_communities[i].extend(group)

                        # Re-assign the rest of the segments from the original community
                        other_segments = [segment_id for segment_id in community if segment_id not in seg_ids_in_frame]

                        for seg_to_assign_id in other_segments:
                            similarities = [self._get_widest_path_similarity(seg_to_assign_id, base_seg_id, similarity_cache) for base_seg_id in base_segments]

                            if sum(similarities) > 0:
                                best_community_idx = np.argmax(similarities)
                                newly_formed_communities[best_community_idx].append(seg_to_assign_id)

                        communities_after_split.extend(new_comm for new_comm in newly_formed_communities if new_comm)

                        # Break from the loop over frames, as this community has been split
                        break

                if not violation_found_in_community:
                    communities_after_split.append(community)

            # After checking all communities, update the list for the next iteration
            final_communities = communities_after_split

            if not a_split_was_made:
                break # No splits in a full pass, exit the while loop.

        self.communities = [[self.node_id_node_idx(node_id) for node_id in community] for community in final_communities]
        return final_communities

    def populateGraph(self,
                      segment_ids: List[str]) -> None:
        filter_conf_min: float = wOmniGlue.MATCH_THRESHOLD      # low threshold for island filter neighbors
        edge_conf_min: float = 0.7                              # high threshold for final edges
        min_filter_matches: int = 2                             # require at least this many >= filter_conf_min
                                                                # between two segments to treat them as neighbors
                                                                # during filtering
        # Square conf thresholds to make them consistent with womniglue.
        filter_conf_min **= 2
        edge_conf_min **= 2

        assert filter_conf_min <= edge_conf_min, f"{filter_conf_min=} should not exceed {edge_conf_min=}."

        # Populate maps
        self.segment_id_to_frame_id_map: Dict[str, str] = {}
        self.keypoint_id_to_segment_id_map: Dict[str, str] = {}
        self.segment_id_to_area_map: Dict[str, int] = {}

        num_keypoints_per_segment: DefaultDict[str, int] = defaultdict(int)

        for segment_id in segment_ids:
            segment = self.segment_manager.get(segment_id, optional=False)
            assert segment is not None
            self.segment_id_to_frame_id_map[segment_id] = segment.parent_frame_id
            self.segment_id_to_area_map[segment_id] = segment.area
            for kp_id in segment.keypoint_ids:
                self.keypoint_id_to_segment_id_map[kp_id] = segment_id
                num_keypoints_per_segment[segment_id] += 1

        frame_ids: List[str] = list(set(self.segment_id_to_frame_id_map.values()))

        # --- Accumulators for edges (high threshold) ---
        frame_id_pairs: List[Tuple[str, str]] = list(itertools.combinations(frame_ids, 2))
        prob_sum_of_confidence: DefaultDict[FrozenSet[str], float] = defaultdict(float)
        sum_of_confidence: DefaultDict[FrozenSet[str], float] = defaultdict(float)
        num_matches: DefaultDict[FrozenSet[str], int] = defaultdict(int)
        prob_sum = lambda c1, c2: 1 - (1 - c1) * (1 - c2)
        num_keypoints_per_pair: DefaultDict[FrozenSet[str], int] = defaultdict(int)

        # --- Accumulators for filter level neighbors (low threshold) ---
        filter_num_matches: DefaultDict[FrozenSet[str], int] = defaultdict(int)
        neighbors_by_segment_and_target_frame: DefaultDict[
            str, DefaultDict[str, Set[str]]
        ] = defaultdict(lambda: defaultdict(set))

        # --- Traverse matches frame-pair-wise ---
        for frame_id_pair in frame_id_pairs:
            if matches := self.matches_manager.get_by_frame_pair(
                frame1_id=frame_id_pair[0],
                frame2_id=frame_id_pair[1],
                optional=True
            ):
                for m in matches:
                    c = m.confidence
                    if c < filter_conf_min:
                        continue

                    kp1_id, kp2_id = m.keypoint_ids
                    segment1_id = self.keypoint_id_to_segment_id_map.get(kp1_id)
                    segment2_id = self.keypoint_id_to_segment_id_map.get(kp2_id)
                    if not segment1_id or not segment2_id:
                        continue

                    if segment1_id == segment2_id:
                        continue

                    pair: FrozenSet[str] = frozenset((segment1_id, segment2_id))

                    # (1) Filter-level evidence: count_matches >= filter_conf_min
                    filter_num_matches[pair] += 1

                    if filter_num_matches[pair] >= min_filter_matches:
                        # When confident enough, register these as neighbors
                        frame1_id = self.segment_id_to_frame_id_map[segment1_id]
                        frame2_id = self.segment_id_to_frame_id_map[segment2_id]
                        neighbors_by_segment_and_target_frame[segment1_id][frame2_id].add(segment2_id)
                        neighbors_by_segment_and_target_frame[segment2_id][frame1_id].add(segment1_id)

                    # (2) Edge-level evidence: only matches >= edge_conf_min
                    if c >= edge_conf_min:
                        prob_sum_of_confidence[pair] = prob_sum(
                            prob_sum_of_confidence[pair], c
                        )
                        sum_of_confidence[pair] += c
                        num_matches[pair] += 1

        # Compute num_keypoints_per_pair only for pairs that have edge-level matches
        for segment_id_pair in num_matches:
            segment1_id, segment2_id = tuple(segment_id_pair)
            num_keypoints_per_pair[segment_id_pair] = (
                num_keypoints_per_segment[segment1_id]
                + num_keypoints_per_segment[segment2_id]
            )

        # --- Create igraph nodes ---
        self.add_nodes([
            {
                "id": segment_id,
                "frame_id": self.segment_id_to_frame_id_map[segment_id],
                "area": self.segment_id_to_area_map[segment_id]
            }
            for segment_id in self.segment_id_to_frame_id_map.keys()
        ])

        # --- Create edges from high-threshold accumulators ---
        edges: List[Dict[str, Union[float, Tuple[str, str]]]] = []
        for pair, prob_sum_of_confidence_single in prob_sum_of_confidence.items():
            if prob_sum_of_confidence_single <= 0:
                continue

            (segment1_id, segment2_id) = tuple(pair)
            area1 = max(int(self.segment_id_to_area_map[segment1_id]), 1)
            area2 = max(int(self.segment_id_to_area_map[segment2_id]), 1)

            configuration = CONFIGURATION

            if configuration == 1:
                w = min(area1, area2) / max(area1, area2)
                confirmation_bonus = np.log(num_matches[pair] + 1)
            elif configuration == 2:
                w = (1.0 / np.sqrt(area1 * area2)) * (
                    (2.0 * area1 * area2) / (area1**2 + area2**2)
                )
                confirmation_bonus = num_matches[pair]
            elif configuration == 3:
                w = 2 / num_keypoints_per_pair[pair]
                confirmation_bonus = num_matches[pair]
            elif configuration == 4:
                w = (2 / num_keypoints_per_pair[pair]) ** 2
                confirmation_bonus = (num_matches[pair]) ** 2
            else:
                raise RuntimeError(
                    f"Configuration can either be 1, 2, 3 or 4 but it is {configuration}."
                )

            weight = prob_sum_of_confidence_single * confirmation_bonus * w

            if weight < 0 :
                continue

            edges.append(
                {
                    "node_ids": (segment1_id, segment2_id),
                    self.sim_attr: weight,
                    "prob_sum_of_confidence": prob_sum_of_confidence_single,
                    "min_over_max_area": w,
                    "confirmation_bonus": confirmation_bonus,
                    "num_of_matches": num_matches[pair],
                    "sum_of_confidence": sum_of_confidence[pair],
                }
            )

        # No parallel edges
        assert len(edges) == len(
            {frozenset(edge["node_ids"]) for edge in edges}  # type: ignore
        ), "Parallel edges are detected unexpectedly."

        # --- Single Mask Island per Frame Rule (using filter-level neighbors) ---
        intersection_cache: Dict[FrozenSet[str], bool] = {}
        bad_segments: Set[str] = set()

        for src_seg_id, neighbors_by_frame in neighbors_by_segment_and_target_frame.items():
            if src_seg_id in bad_segments:
                continue

            for tgt_frame_id, tgt_seg_ids_set in neighbors_by_frame.items():
                tgt_seg_ids = list(tgt_seg_ids_set)

                # Need at least 2 neighbors in the same frame to form >1 island.
                if len(tgt_seg_ids) <= 1:
                    continue

                # Build touching graph over tgt_seg_ids using intersects()
                adj: DefaultDict[str, List[str]] = defaultdict(list)
                for i, s1_id in enumerate(tgt_seg_ids):
                    for s2_id in tgt_seg_ids[i + 1:]:
                        pair_key = frozenset({s1_id, s2_id})
                        if pair_key in intersection_cache:
                            intersects = intersection_cache[pair_key]
                        else:
                            seg1 = self.segment_manager.get(s1_id, optional=False)
                            assert seg1 is not None
                            seg2 = self.segment_manager.get(s2_id, optional=False)
                            assert seg2 is not None
                            intersects = seg1.intersects(seg2)
                            intersection_cache[pair_key] = intersects

                        if intersects:
                            adj[s1_id].append(s2_id)
                            adj[s2_id].append(s1_id)

                # Count connected components among tgt_seg_ids
                visited: Set[str] = set()
                num_components = 0
                violation_here = False

                for s_id in tgt_seg_ids:
                    if s_id not in visited:
                        num_components += 1
                        if num_components > 1:
                            # neighbors in this frame form more than one island -> violation.
                            bad_segments.add(src_seg_id)
                            violation_here = True
                            break

                        stack = [s_id]
                        visited.add(s_id)
                        while stack:
                            u = stack.pop()
                            for nb in adj.get(u, []):
                                if nb not in visited:
                                    visited.add(nb)
                                    stack.append(nb)

                if violation_here:
                    # No need to check other frames for this src_seg_id
                    break

        if bad_segments:
            logger.info(
                f"      Filtering edges incident to {len(bad_segments)} segments "
                f"that connect to multiple mask islands in the same frame."
            )

        # Filter out all edges touching any bad segment
        filtered_edges: List[Dict[str, Union[float, Tuple[str, str]]]] = []
        for e in edges:
            s1, s2 = e["node_ids"]  # type: ignore
            if s1 in bad_segments or s2 in bad_segments:
                continue
            filtered_edges.append(e)

        self.add_edges(filtered_edges)


class SegmentGraph_singleAnyThing(SegmentGraph):
    def __init__(self,
                 segment_ids: List[str],
                 segment_manager: FrameSegmentDescriptorCacheManager,
                 matches_manager: MatchesCacheManager,
                 sim_attr: Optional[str] = None):
        if sim_attr is not None:
            super().__init__(segment_ids=segment_ids,
                             segment_manager=segment_manager,
                             matches_manager=matches_manager,
                             sim_attr=sim_attr)
        else:
            super().__init__(segment_ids=segment_ids,
                             segment_manager=segment_manager,
                             matches_manager=matches_manager)

    def populateGraph(self,
                      segment_ids: List[str]) -> None:
        super().populateGraph(segment_ids)

        edges_to_update: List[Dict[str, Any]] = []
        for edge in self.g.es:
            attrs = edge.attributes()

            node1_idx = edge.source
            node2_idx = edge.target

            if self.node_id_node_idx(node1_idx) is None or self.node_id_node_idx(node2_idx) is None:
                continue

            node1_area, node2_area = self.g.vs["area"][node1_idx], self.g.vs["area"][node2_idx]

            num_matches = attrs.get("num_of_matches")
            sum_of_confidence = attrs.get("sum_of_confidence")

            # Ensure attributes exist and num_matches is not zero
            if num_matches is None or sum_of_confidence is None or num_matches == 0:
                continue

            # Calculate the new similarity score
            new_sim_score = (min(node1_area, node2_area) / max(node1_area, node2_area)) * (sum_of_confidence / num_matches)

            # Get the node IDs for this edge to pass to change_edges
            source_node_id = self.node_id_node_idx(edge.source)
            target_node_id = self.node_id_node_idx(edge.target)

            if source_node_id is None or target_node_id is None:
                continue

            edges_to_update.append({
                "node_ids": (source_node_id, target_node_id),
                self.sim_attr: new_sim_score
            })

        # Apply the changes if there are any
        if edges_to_update:
            self.change_edges(edges_to_update)


class FrameSegmentDescriptor(FrameDescriptorBase):
    """
    Segment Descriptor operating over a rectangular ROI in the original frame.
    """
    def __init__(self,
                 *,
                 parent: FrameDescriptor,
                 parent_coord_mask: np.ndarray,
                 predicted_iou: float,
                 is_dirty: bool = True,
                 check_for_enough_detail: bool = True):
        segment_id = str(uuid.uuid4())
        super().__init__(
            video_container=parent.videoContainer,
            frame_idx=parent.frame_idx,
            womniglue=None,
            kp_manager=parent.kp_manager if hasattr(parent, "kp_manager") else None,
            max_dim_for_processing=parent._max_dim_for_processing,
            max_keypoints=parent._max_keypoints if hasattr(parent, "_max_keypoints") else 0,
            is_dirty=is_dirty,
            id=segment_id
        )

        self.parent_frame_id: str = parent.id
        self.predicted_iou: float = predicted_iou
        self._keypoint_density_threshold: float = parent._keypoint_density_threshold if hasattr(parent, "_keypoint_density_threshold") else 0
        self._parent_scale_cropped_mask: np.ndarray = np.empty(1)
        self._parent_preprocessing_params = parent.preprocessing_params
        self._DINODescirptors = np.array([])
        self._parent_proc_bbox: Optional[Tuple[int, int, int, int]] = None
        self._full_frame_crop: Optional[np.ndarray] = None
        self._check_for_enough_detail: bool = check_for_enough_detail

        # Perform initialization
        self._recompute_from_mask(parent, parent_coord_mask)

    def _recompute_from_mask(self,
                             parent: FrameDescriptor,
                             parent_coord_mask: np.ndarray) -> None:
        """
        (Re)computes the ROI bbox and filters keypoints based on the mask.
        """
        # ---- 1. Calculate the roi_bbox from parent_coord_mask
        contours, _ = cv2.findContours(parent_coord_mask.astype(np.uint8),
                                       cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            raise ValueError("parent_coord_mask is empty and has no contours.")

        self.parent_coord_mask = parent_coord_mask

        (_, _), scale_w, scale_h = self._parent_preprocessing_params
        x_proc, y_proc, w_proc, h_proc = cv2.boundingRect(np.concatenate(contours))

        self._parent_proc_bbox = (x_proc, y_proc, w_proc, h_proc)

        # crop the parent coord mask to the bounding box of the segment.
        self._parent_scale_cropped_mask = parent_coord_mask[y_proc:y_proc+h_proc, x_proc:x_proc+w_proc]

        parent_x0, parent_y0, _, _ = parent.segmentation_source_bbox

        roi_bbox = (
            int(x_proc * scale_w) + parent_x0,
            int(y_proc * scale_h) + parent_y0,
            int(w_proc * scale_w),
            int(h_proc * scale_h)
        )

        x0, y0, w, h = (int(roi_bbox[0]), int(roi_bbox[1]), int(roi_bbox[2]), int(roi_bbox[3]))

        if w <= 0 or h <= 0:
            raise ValueError(f"ROI must have positive size, got {(x0, y0, w, h)}")

        H, W = parent.videoContainer.height, parent.videoContainer.width
        x0 = max(0, min(x0, W - 1))
        y0 = max(0, min(y0, H - 1))
        w = min(w, W - x0)
        h = min(h, H - y0)
        self._roi_bbox: Tuple[int, int, int, int] = (x0, y0, w, h)
        self._full_frame_crop = None
        self._preprocessed_frame = None

        # ---- 2. Filter parent keypoints
        if self._check_for_enough_detail:
            filtered_kp_ids: List[str] = []
            for kp in parent.keypoints:
                # Convert keypoint location to processed frame coordinates
                x_proc_kp = (kp.loc[0] - parent_x0) / scale_w
                y_proc_kp = (kp.loc[1] - parent_y0) / scale_h

                # Adjust coordinates to be relative to the cropped mask
                x_kp_in_mask = x_proc_kp - x_proc
                y_kp_in_mask = y_proc_kp - y_proc

                # Check if the keypoint is within the bounds of the cropped mask
                if 0 <= y_kp_in_mask < self._parent_scale_cropped_mask.shape[0] and 0 <= x_kp_in_mask < self._parent_scale_cropped_mask.shape[1]:
                    # Check if the keypoint is within the mask itself
                    if self._parent_scale_cropped_mask[int(y_kp_in_mask), int(x_kp_in_mask)]:
                        filtered_kp_ids.append(kp.id)

            self._keypoint_ids = tuple(filtered_kp_ids)

        # ---- 3. Invalidate any other caches that depend on keypoints
        self._DINODescriptors = None

    def set_mask(self, parent_coord_mask: np.ndarray, parent: FrameDescriptor) -> None:
        """
        Safely updates the segment's mask and recomputes all dependent properties.
        """
        self._recompute_from_mask(parent, parent_coord_mask)
        self.is_dirty = True

    @property
    def preprocessing_params(self) -> Tuple[Tuple[int, int], float, float]:
        """
        Returns:
            ((proc_h, proc_w), scale_w, scale_h)
        scale_* map processed-ROI coords -> original-image coords.
        """
        _, _, w, h = self._roi_bbox

        if max(h, w) <= self._max_dim_for_processing:
            proc_h, proc_w = h, w
        else:
            s = self._max_dim_for_processing / float(max(h, w)) if max(h, w) > 0 else 0
            proc_w, proc_h = int(round(w * s)), int(round(h * s))

        scale_w = w / float(proc_w) if proc_w else 1.0
        scale_h = h / float(proc_h) if proc_h else 1.0
        return (proc_h, proc_w), scale_w, scale_h

    def _preprocess_frame(self) -> np.ndarray:
        """
        Crop the ROI from the original frame; resize to (proc_w, proc_h) if needed.
        """
        x0, y0, w, h = self._roi_bbox
        (proc_h, proc_w), _, _ = self.preprocessing_params
        frame_full = self.videoContainer._get_frame(self.frame_idx)
        roi = frame_full[y0: y0 + h, x0: x0 + w]
        if roi.shape[0] == proc_h and roi.shape[1] == proc_w:
            return roi
        return cv2.resize(roi, (proc_w, proc_h), interpolation=cv2.INTER_AREA)

    def _preprocessed_to_original_coords(self, kpts_processed: np.ndarray) -> np.ndarray:
        (_, _), scale_w, scale_h = self.preprocessing_params
        x0, y0, _, _ = self._roi_bbox
        out = np.empty_like(kpts_processed, dtype=np.float32)
        out[:, 0] = kpts_processed[:, 0] * scale_w + x0
        out[:, 1] = kpts_processed[:, 1] * scale_h + y0
        # clip to bounds
        np.clip(out[:, 0], 0, self.videoContainer.width - 1, out=out[:, 0])
        np.clip(out[:, 1], 0, self.videoContainer.height - 1, out=out[:, 1])
        return out

    def _original_to_preprocessed_coords(self, kp_locs_orig: np.ndarray) -> np.ndarray:
        """
        Convert original-image keypoints to processed-ROI float coordinates.
        """
        (_, _), scale_w, scale_h = self.preprocessing_params
        x0, y0, _, _ = self._roi_bbox

        x_proc = (kp_locs_orig[:, 0] - x0) / scale_w if scale_w else (kp_locs_orig[:, 0] - x0)
        y_proc = (kp_locs_orig[:, 1] - y0) / scale_h if scale_h else (kp_locs_orig[:, 1] - y0)

        return np.stack([x_proc, y_proc], axis=1)

    def _original_to_preprocessed_indices(self,
                                          kp_locs_orig: np.ndarray,
                                          proc_h: int,
                                          proc_w: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert original-image keypoints to processed-ROI integer indices.
        Return order is (x_idx, y_idx); dense maps are indexed as [y_idx, x_idx].
        """
        kpts_processed = self._original_to_preprocessed_coords(kp_locs_orig)
        x_proc = kpts_processed[:, 0]
        y_proc = kpts_processed[:, 1]

        x_idx = np.rint(x_proc).astype(np.int32)
        y_idx = np.rint(y_proc).astype(np.int32)

        np.clip(x_idx, 0, max(proc_w - 1, 0), out=x_idx)
        np.clip(y_idx, 0, max(proc_h - 1, 0), out=y_idx)
        return x_idx, y_idx

    @property
    def SPFeatures(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
        """
        Return (kpts, desc, scores, kp_ids) on the preprocessed frame, derived from keypoints.
        kpts are in preprocessed-frame coords.
        """
        locs_orig = []
        sp_descriptors_list = []
        scores_list = []
        kp_ids = []

        for kp in self.keypoints:
            locs_orig.append(kp.loc)
            sp_descriptors_list.append(kp.sp_descriptor)
            scores_list.append(kp.score)
            kp_ids.append(kp.id)

        if not locs_orig:
            return (np.empty((0, 2), dtype=np.float32),
                    np.empty((0, 256), dtype=np.float32),
                    np.empty((0,), dtype=np.float32),
                    [])

        kp_locs_orig_arr = np.array(locs_orig)
        sp_descriptors_arr = np.array(sp_descriptors_list)
        scores_arr = np.array(scores_list)

        kpts_processed = self._original_to_preprocessed_coords(kp_locs_orig_arr)

        return (kpts_processed, sp_descriptors_arr, scores_arr, kp_ids)

    @property
    def parent_scale_cropped_mask(self) -> np.ndarray:
        """
        Returns the cropped mask in scaling of the parent and the size of the minimum crop that fits the segment.
        """
        return self._parent_scale_cropped_mask

    @property
    def center_orig_coord(self) -> Optional[Tuple[float, float]]:
        """
        Calculates and returns the center of the mask in the original frame's coordinates.
        """
        moments = cv2.moments(self.full_frame_mask.astype(np.uint8))
        if moments["m00"] != 0:
            cX = moments["m10"] / moments["m00"]
            cY = moments["m01"] / moments["m00"]
            return (cX, cY)
        else:
            return None

    @property
    def full_frame_mask(self) -> np.ndarray:
        x0, y0, w, h = self._roi_bbox
        H, W = self.videoContainer.height, self.videoContainer.width
        _full_frame_mask = np.zeros((H, W), dtype=bool)

        resized_mask = cv2.resize(
            self.parent_scale_cropped_mask.astype(np.uint8),
            (w, h),
            interpolation=cv2.INTER_NEAREST
        ).astype(bool)

        _full_frame_mask[y0: y0+h, x0: x0+w] = resized_mask

        return _full_frame_mask

    @property
    def full_frame_crop(self) -> np.ndarray:
        if self._full_frame_crop is None:
            full_frame = self.videoContainer._get_frame(self.frame_idx)  # HxWx3, RGB
            x0, y0, w, h = self._roi_bbox
            # Crop as [y0:y0+h, x0:x0+w]
            self._full_frame_crop = full_frame[y0: y0 + h, x0: x0 + w].copy()
        assert self._full_frame_crop is not None
        return self._full_frame_crop

    @property
    def enough_detail_captured(self) -> bool:
        """
        Simple keypoint-density gate on the processed_ROI.
        Also checks if the segment's ROI is large enough for DINO processing.
        """
        # The density check is only performed if _check_for_enough_detail is True.
        density_ok = True
        if self._check_for_enough_detail:
            k = len(self.keypoint_ids)
            if self.area > 0:
                density_ok = (k / float(self.area)) >= self._keypoint_density_threshold
            else:
                density_ok = False

        # The DINO processing size check is always performed if dino_patch_size is set.
        dino_size_ok = True
        if hasattr(self, 'dino_patch_size') and self.dino_patch_size is not None:
            (proc_h, proc_w), _, _ = self.preprocessing_params
            dino_size_ok = (proc_w >= self.dino_patch_size) and (proc_h >= self.dino_patch_size)
            # If DINO is used, both size and density (if enabled) must be okay.
            return dino_size_ok and density_ok
        else:
            # If DINO is not used, only the density check (if enabled) matters.
            return density_ok

    @property
    def area(self) -> int:
        """
        Returns the number of pixels that the _parent_scale_cropped_mask covers.
        """
        return np.count_nonzero(self._parent_scale_cropped_mask) # type: ignore

    @property
    def original_frame_area(self) -> int:
        """
        Returns the number of pixels that the segment covers in the original frame.
        """
        return np.count_nonzero(self.full_frame_mask) # type: ignore

    @property
    def DINODescriptorsRowVec(self) -> np.ndarray:
        """
        Collects DINO descriptors from the KeyPoint objects associated with this segment.
        """
        all_descriptors = []
        for kp in self.keypoints:
            if kp.dino_descriptor is not None:
                all_descriptors.append(kp.dino_descriptor.reshape(1, -1)) # Ensure row vector

        if not all_descriptors:
            dino_feature_dim = wOmniGlue.DINO_FEATURE_DIM
            return np.empty((0, dino_feature_dim), dtype=np.float32)

        # Concatenate all collected descriptor arrays vertically.
        return np.concatenate(all_descriptors, axis=0)

    def isolate_disconnected_mask_islands(self, parent: FrameDescriptor) -> None:
        """
        Checks for disconnected components in the mask and updates the mask
        in-place to keeo only the largest, most detailed component.
        """
        mask_uint8 = self.parent_scale_cropped_mask.astype(np.uint8)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_uint8, cv2.CV_32S) # type: ignore

        if num_labels <= 2: # Only background and one component
            return

        best_island_label = -1
        max_detail_ratio = -1.0

        if not self.keypoint_ids:
            # If no keypoints, fall back to the largest area
            best_island_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        else:
            # If there are keypoints, find the island with the best keypoint / area ratio
            kp_locs_orig = np.array([kp.loc for kp in self.keypoints])

            if self._parent_proc_bbox is None:
                raise RuntimeError(
                    "Segment is missing its parent-coordinate bounding box."
                )

            (_, _), scale_w, scale_h = self._parent_preprocessing_params
            parent_x0, parent_y0, _, _ = parent.segmentation_source_bbox
            crop_x, crop_y, _, _ = self._parent_proc_bbox

            # Original image coordinates -> parent-processed cropped-mask coordinates.
            x_mask = (kp_locs_orig[:, 0] - parent_x0) / scale_w - crop_x
            y_mask = (kp_locs_orig[:, 1] - parent_y0) / scale_h - crop_y

            mask_h, mask_w = self.parent_scale_cropped_mask.shape
            valid = (
                (x_mask >= 0) & (x_mask < mask_w)
                & (y_mask >= 0) & (y_mask < mask_h)
            )

            x_indices = x_mask[valid].astype(np.int32)
            y_indices = y_mask[valid].astype(np.int32)

            for i in range(1, num_labels):
                area = stats[i, cv2.CC_STAT_AREA]
                if area == 0:
                    continue

                kp_in_island_indices = np.where(labels[y_indices, x_indices] == i)[0]
                keypoint_count = len(kp_in_island_indices)

                detail_ratio = keypoint_count / area

                if detail_ratio > max_detail_ratio:
                    max_detail_ratio = detail_ratio
                    best_island_label = i

            # Fallback if no island contains keypoints
            if max_detail_ratio <= 0:
                best_island_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])

        if best_island_label != -1:
            island_mask = (labels == best_island_label).astype(np.uint8)

            # Create a new mask in the parent's full preprocessed coordinate system
            proc_h, proc_w = parent.preprocessed_frame.shape[:2]
            new_parent_coord_mask = np.zeros((proc_h, proc_w), dtype=np.uint8)

            # Get the location of the original segment's bounding box
            if self._parent_proc_bbox is None:
                # This should not happen if the object was initialized correctly
                return
            x_p, y_p, w_p, h_p = self._parent_proc_bbox

            # Paste the island mask into the new full-size mask
            new_parent_coord_mask[y_p : y_p + h_p, x_p : x_p + w_p] = island_mask

            self.set_mask(new_parent_coord_mask, parent)

    def calculate_iomin(self, other_segment: "FrameSegmentDescriptor") -> float:
        """
        Calculates the intersection over Minimum Area (IoMin) with another segment
        using their masks.
        """
        mask1 = self.full_frame_mask
        mask2 = other_segment.full_frame_mask

        intersection = np.logical_and(mask1, mask2)

        intersection_area = np.count_nonzero(intersection)

        area1 = self.original_frame_area
        area2 = other_segment.original_frame_area

        min_area = min(area1, area2)

        if min_area == 0:
            return 0.0

        return intersection_area / min_area

    def intersects(self, other_segment: "FrameSegmentDescriptor") -> bool:
        """
        Checks if this segment intersects with another segment.
        """
        # Bounding box check for quick exclusion
        x1, y1, w1, h1 = self._roi_bbox
        x2, y2, w2, h2 = other_segment._roi_bbox
        if (x1 + w1 < x2 or x2 + w2 < x1 or
            y1 + h1 < y2 or y2 + h2 < y1):
            return False

        # Mask intersection check
        mask1 = self.full_frame_mask
        mask2 = other_segment.full_frame_mask

        intersection = np.logical_and(mask1, mask2)

        return bool(np.any(intersection))

    def calculate_intersection_area(self, other_segment: "FrameSegmentDescriptor") -> int:
        """
        Calculates the intersection area in pixels with another segment.
        """
        mask1 = self.full_frame_mask
        mask2 = other_segment.full_frame_mask

        intersection = np.logical_and(mask1, mask2)

        return np.count_nonzero(intersection) # type: ignore

    def to_dict(self) -> Dict[str, Any]:
        data = super().to_dict()
        data.update({
            "__class__": "FrameSegmentDescriptor",
            "parent_frame_id": self.parent_frame_id,
            "predicted_iou": self.predicted_iou,
            "keypoint_density_threshold": self._keypoint_density_threshold,
            "parent_scale_cropped_mask": self._parent_scale_cropped_mask,
            "parent_preprocessing_params": self._parent_preprocessing_params,
            "parent_proc_bbox": self._parent_proc_bbox,
            "roi_bbox": self._roi_bbox,
            "parent_coord_mask": self.parent_coord_mask,
            "full_frame_crop": self._full_frame_crop,
            "check_for_enough_detail": self._check_for_enough_detail
        })
        return data

    @classmethod
    def from_dict(cls: type[T_Seg],
                  data: Dict[str, Any],
                  videoContainer: VideoContainer,
                  kp_manager: "KeyPointCacheManager") -> T_Seg:
        instance = cls.__new__(cls)

        super(FrameSegmentDescriptor, instance).__init__(
            video_container=videoContainer,
            frame_idx=data.get("frame_idx"), # type: ignore
            womniglue=None,
            id=data.get("id"),
            kp_manager=kp_manager,
            max_dim_for_processing=data.get("cache_max_dim_for_processing"),
            max_keypoints=data.get("cache_max_keypoints"),
            is_dirty=False
        )

        instance._keypoint_density_threshold = data["keypoint_density_threshold"]
        instance.parent_frame_id = data["parent_frame_id"]
        instance._parent_scale_cropped_mask = data["parent_scale_cropped_mask"]
        instance._full_frame_crop = data["full_frame_crop"]
        instance._check_for_enough_detail = data["check_for_enough_detail"]

        instance._roi_bbox = tuple(data["roi_bbox"])
        instance.predicted_iou = data["predicted_iou"]
        instance._parent_proc_bbox = data.get("parent_proc_bbox")

        instance._parent_preprocessing_params = data.get("parent_preprocessing_params")

        instance.parent_coord_mask = data.get("parent_coord_mask")

        cls._rehydrate_caches(instance, data, kp_manager)

        instance.is_dirty = False
        return instance


class FrameSegmentDescriptorCacheManager(BaseCacheManager[FrameSegmentDescriptor]):
    def __init__(self,
                 cache_dir: str,
                 videoContainer: VideoContainer,
                 kp_manager: KeyPointCacheManager,
                 use_cache_disk: bool = True,
                 max_size: int = 100):
        super().__init__(cache_dir, FrameSegmentDescriptor, use_cache_disk=use_cache_disk, max_size=max_size)
        self.videoContainer = videoContainer
        self.kp_manager = kp_manager

    def _get_filepath(self, obj_id: str) -> str:
        return os.path.join(self.cache_dir, f"{obj_id}.pkl")

    def _load_from_file(self, filepath: str) -> FrameSegmentDescriptor:
        with open(filepath, "rb") as f:
            data = pickle.load(f)

        return FrameSegmentDescriptor.from_dict(
            data,
            self.videoContainer,
            self.kp_manager
        )
