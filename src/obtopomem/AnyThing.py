# Copyright 2026 Ata Baran Fırat
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import pickle
import uuid
import numpy as np
import cv2
from collections import defaultdict
from typing import List, Dict, Any, Optional, Tuple, FrozenSet, Set, DefaultDict
import itertools

from .BaseCacheManager import BaseCacheManager, Cacheable
from .FrameSegmentDescriptor import FrameSegmentDescriptorCacheManager, SegmentGraph, SegmentGraph_singleAnyThing, TemporaryMatch
from .ImagePoint import KeyPointsCacheManager
from .VideoContainer import VideoContainer
from .Matches import MatchesCacheManager
from .Graph import Graph
from .wOmniGlue import wOmniGlue


class AnyThingGraph(Graph):
    def __init__(self,
                 anything_ids: List[str],
                 anything_manager: AnyThingCacheManager,
                 matches_manager: MatchesCacheManager,
                 segment_manager: FrameSegmentDescriptorCacheManager,
                 segment_graph: SegmentGraph,
                 sim_attr: str = "weight",
                 cache_dir: Optional[str] = None):
        self.anything_manager: AnyThingCacheManager = anything_manager
        self.matches_manager: MatchesCacheManager = matches_manager
        self.segment_manager: FrameSegmentDescriptorCacheManager = segment_manager

        self.cache_dir: Optional[str] = cache_dir

        super().__init__(sim_attr=sim_attr,
                         directed=False,
                         anything_ids=anything_ids,
                         segment_graph=segment_graph)

    @property
    def anything_communities(self) -> List[List[str]]:
        """
        Greedy AnyThing-community merging:
        - Start with each AnyThing as its own community.
        - Repeatedly merge the pair of communities connected by the largest edge weight.
        - After each merge:
            * Recompute edges from the new community to others using best-pair similarity.
            * Enforce the 'must intersect in all common frames' rule at the community level.
        - Stop when no positive-weight edges remain.

        Returns:
            List[List[str]]: Communities of AnyThing IDs.
        """

        # Handle trivial cases
        if not getattr(self, "anything_ids", None):
            self.communities = []
            return []

        if len(self.anything_ids) == 1:
            only = self.anything_ids[0]
            idx = self.node_id_node_idx(only)
            self.communities = [[int(idx)]] if idx is not None else []
            return [[only]]

        # ------------------------------------------------------------
        # Helper closures
        # ------------------------------------------------------------

        def community_frame_ids(comm: set[str]) -> set[str]:
            frames = set()
            for at_id in comm:
                frames |= self.at_id_to_frame_ids_map[at_id]
            return frames

        def community_mask(comm: set[str], frame_id: str) -> np.ndarray:
            # Take any AnyThing to get H, W
            example_at_id = next(iter(comm))
            example_at = self.anything_manager.get(example_at_id, optional=False)
            assert example_at is not None
            H, W = example_at.videoContainer.height, example_at.videoContainer.width

            combined = np.zeros((H, W), dtype=bool)

            for at_id in comm:
                # Skip AnyThings that don't have masks in this frame
                if frame_id not in self.at_id_to_frame_ids_map[at_id]:
                    continue

                at = self.anything_manager.get(at_id, optional=False)
                assert at is not None

                mask = at.full_frame_mask(frame_id)
                combined |= mask

            return combined

        def communities_intersect_all_frames(commA: set[str], commB: set[str]) -> bool:
            fA = community_frame_ids(commA)
            fB = community_frame_ids(commB)
            common = fA & fB
            if not common:
                return True
            for frame in common:
                if not np.any(community_mask(commA, frame) &
                              community_mask(commB, frame)):
                    return False
            return True

        def best_edge_weight(commA: set[str], commB: set[str]) -> float:
            best = 0.0
            for a in commA:
                for b in commB:
                    w = self.get_edge_parameter((a, b), default=0)
                    if w > best:
                        best = w
            return best

        def communities_segment_ids(comms: Dict[int, Set[str]]) -> List[List[str]]:
            result = []
            for at_ids in comms.values():
                one_comm = []
                for at_id in at_ids:
                    one_comm.extend(self.at_id_to_seg_ids_map[at_id])
                result.append(one_comm)
            return result

        # ------------------------------------------------------------
        # Initialize each AnyThing as a community
        # ------------------------------------------------------------

        communities: Dict[int, Set[str]] = {}
        next_id = 0
        for at in self.anything_ids:
            communities[next_id] = {at}
            next_id += 1

        # ------------------------------------------------------------
        # Build initial adjacency (community → {neighbor → weight})
        # ------------------------------------------------------------

        from collections import defaultdict as _ddict
        adj: Dict[int, Dict[int, float]] = _ddict(dict)

        comm_ids = list(communities.keys())
        for i, ci in enumerate(comm_ids):
            for cj in comm_ids[i+1:]:
                A, B = communities[ci], communities[cj]
                w = best_edge_weight(A, B)
                if w > 0 and communities_intersect_all_frames(A, B):
                    adj[ci][cj] = w
                    adj[cj][ci] = w

        # ------------------------------------------------------------
        # Greedy merging loop
        # ------------------------------------------------------------

        count = 0
        if self.cache_dir is not None:
                self.segment_graph.visualize_graph_view(save_folder=self.cache_dir+f"/AnyThingGraphMerges_on_SegmentGraph_{count}.png",
                                                        communities_in=communities_segment_ids(communities))
        while True:
            # Find strongest available community edge
            best_w = 0.0
            best_pair = None

            for ci, neighbors in adj.items():
                if ci not in communities:
                    continue
                for cj, w in neighbors.items():
                    if cj not in communities:
                        continue
                    if w > best_w:
                        best_w = w
                        best_pair = (ci, cj)

            # Stop if no mergeable edge
            if best_pair is None or best_w <= 0.0:
                break

            ci, cj = best_pair
            # These may have been removed in earlier loops
            if ci not in communities or cj not in communities or ci == cj:
                continue

            # Merge communities
            new_set = communities[ci] | communities[cj]
            del communities[ci]
            del communities[cj]

            new_id = next_id
            next_id += 1
            communities[new_id] = new_set

            # Remove old adjacency references
            adj.pop(ci, None)
            adj.pop(cj, None)
            for nbrs in adj.values():
                nbrs.pop(ci, None)
                nbrs.pop(cj, None)

            # Add adjacency entries for the new community
            adj[new_id] = {}
            for ck, commK in list(communities.items()):
                if ck == new_id:
                    continue
                w = best_edge_weight(new_set, commK)
                if w > 0 and communities_intersect_all_frames(new_set, commK):
                    adj[new_id][ck] = w
                    adj[ck][new_id] = w

            count += 1
            if self.cache_dir is not None and count % 5 == 0:
                self.segment_graph.visualize_graph_view(save_folder=self.cache_dir+f"/AnyThingGraphMerges_on_SegmentGraph_{count}.png",
                                                        communities_in=communities_segment_ids(communities))

        # ------------------------------------------------------------
        # Convert communities to lists of IDs
        # ------------------------------------------------------------

        result: List[List[str]] = []
        for comm in communities.values():
            result.append(sorted(comm))

        # Also populate Graph.communities for visualization (vertex indices)
        vertex_comms: List[List[int]] = []
        for comm_ids in result:
            indices = []
            for at in comm_ids:
                idx = self.node_id_node_idx(at)
                if idx is not None:
                    indices.append(int(idx))
            if indices:
                vertex_comms.append(indices)

        self.communities = vertex_comms

        return result

    def populateGraph(self,
                      anything_ids: List[str],
                      segment_graph: SegmentGraph) -> None:
        min_number_of_edges = 2
        self.anything_ids: List[str] = []

        self.segment_graph: SegmentGraph = segment_graph

        self.kp_id_to_seg_id_map: Dict[str, str] = {}
        self.at_id_to_seg_ids_map: DefaultDict[str, List[str]] = defaultdict(list)
        self.at_id_to_frame_ids_map: DefaultDict[str, set[str]] = defaultdict(set)
        self.all_frame_ids = set()

        # Populate maps
        for at_id in anything_ids:
            anything = self.anything_manager.get(at_id, optional=False)
            assert anything is not None
            self.anything_ids.append(at_id)
            self.at_id_to_frame_ids_map[at_id] |= set(anything.frame_ids_with_mask)
            for seg_id in anything.segment_ids:
                seg = self.segment_manager.get(seg_id, optional=False)
                assert seg is not None
                self.at_id_to_seg_ids_map[at_id].append(seg_id)
                self.all_frame_ids.add(seg.parent_frame_id)
                for kp_id in seg.keypoint_ids:
                    self.kp_id_to_seg_id_map[kp_id] = seg_id

        self.add_nodes([{"id": anything_id} for anything_id in self.anything_ids])

        at_id_pairs = itertools.combinations(anything_ids, 2)

        edges = []
        for at_id_pair in at_id_pairs:
            at1_id, at2_id = at_id_pair

            # Check for common frames between anythings.
            at1_frame_ids = self.at_id_to_frame_ids_map[at1_id]
            at2_frame_ids = self.at_id_to_frame_ids_map[at2_id]

            common_frames = at1_frame_ids & at2_frame_ids

            at1 = self.anything_manager.get(at1_id, optional=False)
            assert at1 is not None
            at2 = self.anything_manager.get(at2_id, optional=False)
            assert at2 is not None

            # Check if AnyThings do intersect at the common frames.
            intersecting_frames_count = 0
            for common_frame in common_frames:
                at1_mask = at1.full_frame_mask(common_frame)
                at2_mask = at2.full_frame_mask(common_frame)
                if bool(np.any(at1_mask & at2_mask)):
                    intersecting_frames_count += 1

            if intersecting_frames_count < len(common_frames):
                continue

            # Obtain edge weigts from SegmentGraph
            at1_seg_ids = self.at_id_to_seg_ids_map[at1_id]
            at2_seg_ids = self.at_id_to_seg_ids_map[at2_id]

            best_sim = 0
            num_edges = 0
            for seg1_id in at1_seg_ids:
                for seg2_id in at2_seg_ids:
                    sim = segment_graph.get_edge_parameter((seg1_id, seg2_id))
                    if sim > 0:
                        num_edges += 1
                    if sim > best_sim:
                        best_sim = sim

            if num_edges < min_number_of_edges:
                best_sim = 0

            if best_sim > 0:
                edges.append({
                    "node_ids": (at1_id, at2_id),
                    self.sim_attr: best_sim
                })

        self.add_edges(edges)


class AnyThing(Cacheable):
    """
    Represents a collection of segments and keypoints across different frames that are associated
    with the same object instance. Can store multiple segments and keypoints from a single frame.
    """
    def __init__(self,
                 segment_manager: FrameSegmentDescriptorCacheManager,
                 kps_manager: KeyPointsCacheManager,
                 videoContainer: VideoContainer,
                 matches_manager: MatchesCacheManager,
                 initial_segment_ids: Optional[List[str]] = None,
                 id: Optional[str] = None,
                 is_dirty: bool = True,
                 num_representative_segments: int = 5):
        self._id = id if id is not None else str(uuid.uuid4())
        self._is_dirty = is_dirty
        self.segment_manager = segment_manager
        self.kps_manager = kps_manager
        self.matches_manager = matches_manager
        self.videoContainer = videoContainer

        self.num_representative_segments: int = num_representative_segments
        self._representative_segment_ids: Optional[List[str]] = None
        self.segment_graph: Optional[SegmentGraph_singleAnyThing] = None

        # Store segment IDs, keyed by frame_id. Each frame can have multiple segments.
        self.frame_id_to_segment_id_map: Dict[str, List[str]] = defaultdict(list)
        if initial_segment_ids:
            for seg_id in initial_segment_ids:
                self.add_segment(seg_id)

        # Store keypoint cloud IDs, keyed by frame_id. Each frame can have multiple keypoint clouds.
        self.frame_id_to_keypoints_id_map: Dict[str, List[str]] = defaultdict(list)

    @property
    def id(self) -> str:
        return self._id
    @id.setter
    def id(self, new_id) -> None:
        self._id = new_id

    @property
    def is_dirty(self) -> bool:
        return self._is_dirty
    @is_dirty.setter
    def is_dirty(self, new_is_dirty: bool) -> None:
        self._is_dirty = new_is_dirty

    def add_segment(self, segment_id: str):
        """Adds a new segment to the collection."""
        segment = self.segment_manager.get(segment_id, optional=False)
        if segment is None:
            return

        frame_id = segment.parent_frame_id
        if segment.id not in self.frame_id_to_segment_id_map[frame_id]:
            self.frame_id_to_segment_id_map[frame_id].append(segment_id)
            self._representative_segment_ids = None
            self.segment_graph = None
            self.is_dirty = True

    def add_keypoint_cloud(self, kps_id: str):
        """Adds a new KeyPoints object to the collection."""
        kps = self.kps_manager.get(kps_id, optional=False)
        if kps is None:
            return

        frame_id = kps.frame_id
        if kps_id not in self.frame_id_to_keypoints_id_map[frame_id]:
            self.frame_id_to_keypoints_id_map[frame_id].append(kps_id)
            self.is_dirty = True

    @property
    def segment_ids(self) -> List[str]:
        return [segment_id for segment_ids in self.frame_id_to_segment_id_map.values() for segment_id in segment_ids]

    @property
    def keypoints_ids(self) -> List[str]:
        return [keypoints_id for keypoints_ids in self.frame_id_to_keypoints_id_map.values() for keypoints_id in keypoints_ids]

    @property
    def frame_ids_with_mask(self) -> List[str]:
        return list(self.frame_id_to_segment_id_map.keys())

    @property
    def frame_ids_with_keypoints(self) -> List[str]:
        return list(self.frame_id_to_keypoints_id_map.keys())

    @property
    def frame_ids(self) -> List[str]:
        out = self.frame_ids_with_keypoints
        out.extend(self.frame_ids_with_mask)
        return list(set(out))

    @property
    def representative_segment_ids(self) -> List[str]:
        if self._representative_segment_ids is None:
            self.segment_graph = SegmentGraph_singleAnyThing(
                self.segment_ids,
                self.segment_manager,
                self.matches_manager
            )
            self._representative_segment_ids = self.segment_graph.representative_MMR(self.num_representative_segments)
            self.is_dirty = True
        return self._representative_segment_ids

    def segment_id_in_frame(self, frame_id: str) -> Optional[List[str]]:
        if frame_id in self.frame_ids_with_mask:
            return self.frame_id_to_segment_id_map[frame_id]

    def center_in_frames(self, frame_ids: List[str]) -> List[Tuple[float, float]]:
        centers_per_frame = []
        for frame_id in frame_ids:
            if not (segment_ids := self.segment_id_in_frame(frame_id)):
                centers_per_frame.append((np.nan, np.nan))
                continue

            if len(segment_ids) == 1:
                segment_id = segment_ids[0]
                segment = self.segment_manager.get(segment_id, optional=False)
                assert segment is not None
                center = segment.center_orig_coord
                assert center is not None
                centers_per_frame.append(center)
                continue

            centers = []
            areas = []
            for segment_id in segment_ids:
                segment = self.segment_manager.get(segment_id, optional=False)
                assert segment is not None
                center = segment.center_orig_coord
                area = segment.area
                if center is not None:
                    centers.append(center)
                    areas.append(area)
            center1 = 0
            center2 = 0
            sum_area = 0
            for i, center in enumerate(centers):
                center1 += center[0] * areas[i]
                center2 += center[1] * areas[i]
                sum_area += areas[i]

            if sum_area > 0 and np.isfinite(sum_area):
                centers_per_frame.append((center1 / sum_area, center2 / sum_area))
            else:
                centers_per_frame.append((np.nan, np.nan))

        return centers_per_frame

    def filter_wrt_noOfKeyPoints(self, threshold: float = 3.0) -> None:
        """
        Remove segments that have an unusually large number of keypoints.

        For each frame where this AnyThing has segments:
        - compute the total number of keypoints the AnyThing has in that frame
          (sum over all its segments in that frame),
        - collect these per-frame totals into a list and take the median.

        Any segment whose own number of keypoints exceeds
            threshold * median_per_frame_keypoints
        is removed from this AnyThing.

        After filtering, cached attributes that depend on segments are reset.
        """
        # --- 1. Collect per-frame total keypoint counts ---
        per_frame_kp_counts: List[int] = []

        for frame_id, seg_ids in self.frame_id_to_segment_id_map.items():
            total_kps = 0
            for seg_id in seg_ids:
                seg = self.segment_manager.get(seg_id, optional=False)
                if seg is None:
                    continue
                kps_ids = getattr(seg, "keypoint_ids", [])
                total_kps += len(kps_ids)
            per_frame_kp_counts.append(total_kps)

        # Nothing to do if no keypoints information at all
        if not per_frame_kp_counts:
            return

        median_kps = float(np.median(per_frame_kp_counts))
        # If median is zero or negative (degenerate), don't filter
        if median_kps <= 0:
            return

        cutoff = threshold * median_kps

        # --- 2. Filter segments whose own kp count exceeds cutoff ---
        from collections import defaultdict
        new_frame_id_to_segment_id_map: Dict[str, List[str]] = defaultdict(list)
        any_removed = False

        for frame_id, seg_ids in self.frame_id_to_segment_id_map.items():
            for seg_id in seg_ids:
                seg = self.segment_manager.get(seg_id, optional=False)
                if seg is None:
                    continue
                kps_ids = getattr(seg, "keypoint_ids", [])
                num_kps = len(kps_ids)

                if num_kps > cutoff:
                    any_removed = True
                    # drop this segment
                    continue

                new_frame_id_to_segment_id_map[frame_id].append(seg_id)

        # --- 3. If something changed, update map and reset caches ---
        if any_removed:
            self.frame_id_to_segment_id_map = new_frame_id_to_segment_id_map
            # Invalidate cached / derived structures that depend on segments
            self._representative_segment_ids = None
            self.segment_graph = None
            self.is_dirty = True

    def keypoints_id_in_frame(self, frame_id: str) -> Optional[List[str]]:
        if frame_id in self.frame_ids_with_keypoints:
            return self.frame_id_to_keypoints_id_map[frame_id]

    def full_frame_mask(self, frame_id: str) -> np.ndarray:
        H, W = self.videoContainer.height, self.videoContainer.width

        if not (segment_ids_for_frame := self.segment_id_in_frame(frame_id)):
            print(f"AnyThing object does not contain masks from frame_id: {frame_id}")
            return np.zeros((H, W), dtype=bool)

        # Initialize an empty mask with the correct dimensions.
        combined_mask = np.zeros((H, W), dtype=bool)

        for seg_id in segment_ids_for_frame:
            seg = self.segment_manager.get(seg_id, optional=False)
            if seg is not None:
                combined_mask |= seg.full_frame_mask # Combine mask

        return combined_mask

    def full_frame_keypoints_locs(self, frame_id: str) -> np.ndarray:
        H, W = self.videoContainer.height, self.videoContainer.width

        if not (keypoints_ids_for_frame := self.keypoints_id_in_frame(frame_id)):
            print(f"AnyThing does not contain keypoints from frame_id: {frame_id}")
            return np.zeros((H, W), dtype=bool)

        xs_all, ys_all = [], []

        for kps_id in keypoints_ids_for_frame:
            if kps := self.kps_manager.get(kps_id, optional=False):
                if kps.locs.shape[1] > 0:
                    xs, ys = kps.locs[0], kps.locs[1]
                    xs_all.append(xs)
                    ys_all.append(ys)

        if not xs_all or not ys_all:
            return np.zeros((H, W), dtype=bool)

        xs = np.concatenate(xs_all)
        ys = np.concatenate(ys_all)

        xs = np.rint(xs).astype(int)
        ys = np.rint(ys).astype(int)

        inb = (0 <= xs) & (xs < W) & (0 <= ys) & (ys < H)
        xs, ys = xs[inb], ys[inb]

        combined_points = np.zeros((H, W), dtype=bool)
        combined_points[ys, xs] = True

        return combined_points

    @property
    def DINODescriptorsColVec(self) -> np.ndarray:
        """
        Collects all DINODescriptorsColVec from all segments and keypoints of this
        AnyThing and returns them as a single 2D numpy array where each column is a
        descriptor. Shape will be (DINO Feature Dim, Total Number of KeyPoints).
        """
        dino_feature_dim = 0
        if self.segment_manager:
            dino_feature_dim = wOmniGlue.DINO_FEATURE_DIM

        all_descriptors = []
        for segment_ids_for_frame in self.frame_id_to_segment_id_map.values():
            for seg_id in segment_ids_for_frame:
                # Ensure the segment has the DINODescriptorColVec property
                if not (seg := self.segment_manager.get(seg_id, optional=False)):
                    return np.empty((dino_feature_dim, 0))
                if hasattr(seg, "DINODescriptorsColVec"):
                    descriptors = seg.DINODescriptorsColVec
                    if descriptors.size > 0:
                        all_descriptors.append(descriptors)

        for kps_ids_for_frame in self.frame_id_to_keypoints_id_map.values():
            for kps_id in kps_ids_for_frame:
                if kps := self.kps_manager.get(kps_id, optional=False):
                    if hasattr(kps, "DINODescriptorsColVec"):
                        descriptors = kps.DINODescriptorsColVec
                        if descriptors.size > 0:
                            all_descriptors.append(descriptors)

        if not all_descriptors:
            return np.empty((dino_feature_dim, 0), dtype=np.float32)

        return np.concatenate(all_descriptors, axis=1)

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the AnyThing object."""
        return {"id": self.id,
                "frame_id_to_segment_id_map": self.frame_id_to_segment_id_map,
                "frame_id_to_keypoints_id_map": self.frame_id_to_keypoints_id_map,
                "representative_segment_ids": self._representative_segment_ids,
                "num_representative_segments": self.num_representative_segments}

    @classmethod
    def from_dict(cls,
                  data: Dict[str, Any],
                  segment_manager: FrameSegmentDescriptorCacheManager,
                  kps_manager: KeyPointsCacheManager,
                  matches_manager: MatchesCacheManager,
                  videoContainer: VideoContainer) -> AnyThing:
        # Create new instance without any initial segments
        instance = cls(segment_manager=segment_manager,
                       kps_manager=kps_manager,
                       matches_manager=matches_manager,
                       videoContainer=videoContainer,
                       num_representative_segments=data.get("num_representative_segments", 5),
                       id=data.get("id"),
                       is_dirty=False)

        # Manually set the frame_id_to_segment_id_map from the loaded data.
        segment_map_data = data.get("frame_id_to_segment_id_map")
        instance.frame_id_to_segment_id_map = segment_map_data if segment_map_data is not None else defaultdict(list)

        # Manually set the frame_id_to_keypoints_id_map from the loaded data.
        keypoints_map_data = data.get("frame_id_to_keypoints_id_map")
        instance.frame_id_to_keypoints_id_map = keypoints_map_data if keypoints_map_data is not None else defaultdict(list)

        # Manually set the representative segment ids from the loaded data.
        instance._representative_segment_ids = data.get("representative_segment_ids", None)

        return instance


class AnyThingCacheManager(BaseCacheManager[AnyThing]):
    """A cache manager for AnyThing objects."""
    def __init__(self,
                 cache_dir: str,
                 segment_manager: FrameSegmentDescriptorCacheManager,
                 kps_manager: KeyPointsCacheManager,
                 matches_manager: MatchesCacheManager,
                 videoContainer: VideoContainer,
                 use_cache_disk: bool = True):
        super().__init__(cache_dir, AnyThing, use_cache_disk=use_cache_disk)
        self.segment_manager = segment_manager
        self.kps_manager = kps_manager
        self.matches_manager = matches_manager
        self.videoContainer = videoContainer

    def _get_filepath(self, obj_id: str) -> str:
        return os.path.join(self.cache_dir, f"{obj_id}.pkl")

    def _load_from_file(self, filepath: str) -> AnyThing:
        with open(filepath, "rb") as f:
            data = pickle.load(f)
        return AnyThing.from_dict(data,
                                  segment_manager=self.segment_manager,
                                  kps_manager=self.kps_manager,
                                  matches_manager=self.matches_manager,
                                  videoContainer=self.videoContainer)
