# Copyright 2026 Ata Baran Fırat
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np
from typing import List, Dict, Any, Tuple, Optional, Set, Union, FrozenSet, overload
import os
import pickle
from bidict import bidict
import logging
import math
from tqdm import tqdm

from .VideoContainer import VideoContainer
from .BaseCacheManager import BaseCacheManager, Cacheable
from .ImagePoint import KeyPointCacheManager, KeyPointsCacheManager
from .Matches import Matches, MatchesCacheManager, MatchCacheManager
from .AnyThing import AnyThing, AnyThingCacheManager
from .FrameDescriptor import FrameDescriptor, FrameDescriptorCacheManager
from .FrameSegmentDescriptor import FrameSegmentDescriptor, FrameSegmentDescriptorCacheManager, TemporaryMatch
from .Graph import Graph
from .wSAM import wSAM

logger = logging.getLogger(__name__)

SAM_PRED_IOU_THRESH = 0.95
SAM_STABILITY_SCORE_THRESH = 0.98
SAM_BOX_NMS_THRESH = 0.2
SAM_CROP_NMS_THRESH = 0.05



class TopoEnv(Cacheable):
    """
    Represents the topological environment extracted from a video, managing tracked
    objects (AnyThings) and their spatial relationships across frames.

    This class serves as a central repository for topological data, including pixel
    coordinates for both original and dynamically added AnyThings.
    It integrates with a caching mechanism to optimize computation of these topological maps.
    """
    def __init__(self,
                 cache_dir: str,
                 anything_ids: List[str],
                 preprocessing_params: Tuple[Tuple[int, int], float, float],
                 *,
                 video_path: Optional[str] = None,
                 videoContainer: Optional[VideoContainer] = None,
                 max_dim_for_processing: int = 630,
                 wsam: Optional[wSAM] = None,
                 sam_pred_iou_thresh: float = SAM_PRED_IOU_THRESH,
                 sam_stability_score_thresh: float = SAM_STABILITY_SCORE_THRESH,
                 sam_box_nms_thresh: float = SAM_BOX_NMS_THRESH,
                 sam_crop_nms_thresh: float = SAM_CROP_NMS_THRESH,
                 id: Optional[str] = None,
                 is_dirty: bool = True):
        """
        Initializes a new TopoEnv instance, setting up the environment with initial AnyThings
        and frame data, and preparing internal mappings and caches.
        """
        self._max_dim_for_processing: int = max_dim_for_processing
        self.use_cache_disk: bool = True

        self.cache_dir = cache_dir

        if videoContainer is None:
            if video_path is None:
                raise ValueError(
                    f"Either videoContainer or video_path should be supplied."
                )
            self.videoContainer = VideoContainer(video_path)
        else:
            self.videoContainer = videoContainer

        self.video_id = self.videoContainer.id

        self.wsam: Optional[wSAM] = None
        if wsam is None:
            self.wsam = wSAM(
            pred_iou_thresh=sam_pred_iou_thresh,
            stability_score_thresh=sam_stability_score_thresh,
            box_nms_thresh=sam_box_nms_thresh,
            crop_nms_thresh=sam_crop_nms_thresh
            )
        else:
            self.wsam = wsam

        # Initialize cache managers
        self.kp_manager = KeyPointCacheManager(cache_dir=self.cache_dir,
                                               use_cache_disk=self.use_cache_disk)

        self.kps_manager = KeyPointsCacheManager(cache_dir=self.cache_dir,
                                                 kp_manager=self.kp_manager,
                                                 use_cache_disk=self.use_cache_disk)

        self.match_manager = MatchCacheManager(cache_dir=self.cache_dir,
                                               keypoint_manager=self.kp_manager,
                                               use_cache_disk=self.use_cache_disk)

        self.matches_manager = MatchesCacheManager(cache_dir=self.cache_dir,
                                                   match_manager=self.match_manager,
                                                   use_cache_disk=self.use_cache_disk)

        self.segment_manager = FrameSegmentDescriptorCacheManager(cache_dir=self.cache_dir,
                                                                  videoContainer=self.videoContainer,
                                                                  kp_manager=self.kp_manager,
                                                                  use_cache_disk=self.use_cache_disk)

        self.anything_manager = AnyThingCacheManager(cache_dir=self.cache_dir,
                                                     segment_manager=self.segment_manager,
                                                     kps_manager=self.kps_manager,
                                                     matches_manager=self.matches_manager,
                                                     videoContainer=self.videoContainer,
                                                     use_cache_disk=self.use_cache_disk)

        self.frame_manager = FrameDescriptorCacheManager(cache_dir=self.cache_dir,
                                                         videoContainer=self.videoContainer,
                                                         keypoint_manager=self.kp_manager,
                                                         segment_manager=self.segment_manager,
                                                         womniglue=None,
                                                         wsam=self.wsam,
                                                         use_cache_disk=self.use_cache_disk)

        self.preprocessing_params = preprocessing_params

        self._id: str = self.video_id + "_TopoEnv" if id is None else id

        self._is_dirty: bool = is_dirty

        self.video_size: Tuple[int, int] = self.videoContainer.size
        self._anything_ids_orig: List[str] = anything_ids
        self._anything_ids_added: List[str] = []

        self._anything_id_to_index_map: bidict[str, int] = bidict({
            anything_id: i for i, anything_id in enumerate(self.anything_ids_orig)
        })

        frame_ids: List[str] = []
        self._anything_id_to_frame_ids_map: Dict[str, List[str]] = {}

        for anything_id in self._anything_ids_orig:
            anything = self.anything_manager.get(anything_id, optional=False)
            assert anything is not None
            frame_ids.extend(anything.frame_ids)
            self._anything_id_to_frame_ids_map[anything_id] = anything.frame_ids

        self._frame_ids_orig: List[str] = list(set(frame_ids))
        self._frame_ids_added: List[str] = []

        self._frame_id_to_index_map: bidict[str, int] = bidict({
            frame_id: i for i, frame_id in enumerate(self.frame_ids_orig)
        })

        self._topoManOrigFrames: Optional[np.ndarray] = None
        self._topoManAddedFrames: Optional[np.ndarray] = None
        self._topoManFilled: Optional[np.ndarray] = None

        self._F_map: Optional[Dict[Tuple[int, int], np.ndarray]] = None
        self._sampson_err: Optional[Dict[Tuple[int, int], np.ndarray]] = None

    def invalidateTopoManCache(self) -> None:
        self._topoManOrigFrames = None
        self._topoManAddedFrames = None
        self._topoManFilled = None
        self._F_map, self._sampson_err = None, None
        self.is_dirty = True

    @property
    def id(self) -> str:
        return self._id
    @id.setter
    def id(self, new_id: str) -> None:
        self._id = new_id

    @property
    def is_dirty(self) -> bool:
        return self._is_dirty
    @is_dirty.setter
    def is_dirty(self, new_is_dirty: bool) -> None:
        self._is_dirty = new_is_dirty

    @property
    def anything_ids_orig(self) -> List[str]:
        """
        Gets a list of the original AnyThing IDs that were present when the TopoEnv was initialized.
        """
        return self._anything_ids_orig

    @property
    def anything_ids_added(self) -> List[str]:
        """
        Gets a list of AnyThing IDs that have been added to the TopoEnv since initialization.
        """
        return self._anything_ids_added

    @property
    def anything_ids(self) -> List[str]:
        """
        Gets a list of AnyThing IDs currently managed by the TopoEnv (original + added).
        """
        return self.anything_ids_orig + self.anything_ids_added

    @property
    def frame_ids_orig(self) -> List[str]:
        """
        Gets a list of frame IDs that were present when the TopoEnv was initialized.
        """
        return self._frame_ids_orig

    @property
    def frame_ids_added(self) -> List[str]:
        """
        Gets a list of frame IDs that have been dynamically added to the TopoEnv since initialization.
        """
        return self._frame_ids_added

    @property
    def frame_ids(self) -> List[str]:
        """
        Gets a list of frame IDs currently managed by the TopoEnv (original + added).
        """
        return self.frame_ids_orig + self.frame_ids_added

    @property
    def num_anythings_orig(self) -> int:
        """
        Gets the total number of original AnyThings in the environment.
        """
        return len(self.anything_ids_orig)

    @property
    def num_anythings_added(self) -> int:
        """
        Gets the total number of added AnyThings in the environment.
        """
        return len(self.anything_ids_added)

    @property
    def num_anythings(self) -> int:
        """
        Gets the total number of AnyThings (original + added) in the environment.
        """
        return len(self.anything_ids)

    @property
    def num_frames_orig(self) -> int:
        """
        Gets the total number of original frames in the environment.
        """
        return len(self.frame_ids_orig)

    @property
    def num_frames_added(self) -> int:
        """
        Gets the total number of added frames in the environment.
        """
        return len(self.frame_ids_added)

    @property
    def num_frames(self) -> int:
        """
        Gets the total number of frames (original + added) in the environment.
        """
        return len(self.frame_ids)

    @property
    def topoManOrigFrames(self) -> np.ndarray:
        """
        Gets a 3D numpy array representing the pixel coordinates of all AnyThings
        within the original set of frames.

        Shape: (num_frames, 2, num_anythings)
        The second dimension (2) represents (x,y) coordinates.
        Unobserved values will be NaN.
        """
        if self._topoManOrigFrames is None:
            self._topoManOrigFrames = self.createTopoManPix(frame_type="orig",
                                                            anything_type="all")
        return self._topoManOrigFrames

    @property
    def topoManAddedFrames(self) -> np.ndarray:
        """
        Gets a 3D numpy array representing the pixel coordinates of all AnyThings
        within the added set of frames.

        Shape: (num_frames, 2, num_anythings)
        The second dimension (2) represents (x, y) pixel coordinates.
        Unobserved values will be NaN.
        """
        if self._topoManAddedFrames is None:
            self._topoManAddedFrames = self.createTopoManPix(frame_type="added",
                                                             anything_type="all")
        return self._topoManAddedFrames

    @property
    def topoManAllFrames(self) -> np.ndarray:
        """
        Gets a 3D numpy array representing the pixel coordinates of all AnyThings
        within all frames.

        Shape: (num_frames, 2, num_anythings)
        The second dimension (2) represents (x, y) pixel coordinates.
        Unobserved values will be NaN.
        """
        orig  = self.topoManOrigFrames.astype(float, copy=False)
        added = self.topoManAddedFrames.astype(float, copy=False)

        # --- (1) Check that each (x,y) pair is consistent: both NaN or both finite
        # Compute a mask for "exactly one component is NaN"
        inconsistent_orig = np.isnan(orig[:, 0, :]) ^ np.isnan(orig[:, 1, :])
        inconsistent_added = np.isnan(added[:, 0, :]) ^ np.isnan(added[:, 1, :])

        if np.any(inconsistent_orig):
            bad = np.argwhere(inconsistent_orig)
            raise ValueError(f"Inconsistent (x,y) NaN pattern in topoManPixOrigFrames at indices {bad[:5]}...")

        if np.any(inconsistent_added):
            bad = np.argwhere(inconsistent_added)
            raise ValueError(f"Inconsistent (x,y) NaN pattern in topoManPixAddedFrames at indices {bad[:5]}...")

        # --- (2) Check that both sources don't have valid values simultaneously
        both_valid = (~np.isnan(orig).any(axis=1)) & (~np.isnan(added).any(axis=1))
        if np.any(both_valid):
            bad = np.argwhere(both_valid)
            raise ValueError(f"Conflict: both topoManPixOrigFrames and topoManPixAddedFrames "
                            f"have valid data at indices {bad[:5]}...")

        # --- (3) Merge NaN-aware at the pair level
        pair_missing = np.isnan(orig).any(axis=1)        # (F, A)
        pair_missing = pair_missing[:, None, :]          # (F, 1, A) # type: ignore
        merged = np.where(pair_missing, added, orig)

        return merged

    @property
    def topoManFilled_w_log(self) -> Tuple[np.ndarray, Dict[Tuple[int, int], np.ndarray], Dict[Tuple[int, int], np.ndarray]]:
        """
        Gets geometrically reconstructed topological pixel map together with the estimated Fundamental
        matrices and their Sampson error diagnostics.

        Returns:
            Tuple containing:
                - Filled pixel coordinates of shape: (num_frames, 2, num_anythings)
                - Dictionary mapping frame-index pairs (i, j) to their estimated Fundamental matrices.
                - Dictionary mapping frame-index pairs (i, j) to squared Sampson errors for their common
                    observed correspondences.

        The results are cached until the topological map cache is invalidated.
        """
        if self._topoManFilled is None or self._F_map is None or self._sampson_err is None:
            self._topoManFilled, self._F_map, self._sampson_err = TopoUtils.fill_pix_via_F_w_log(self.topoManAllFrames)
        return self._topoManFilled, self._F_map, self._sampson_err

    @property
    def topoManFilled(self) -> np.ndarray:
        """
        Gets a 3D numpy array containing observed and geometrically reconstructed pixel coordinates
        for all AnyThings across all frames.

        This property is cached and computed by filling NaN values in 'topoManAllFrames'
        using epipolar geometry.
        Shape: (num_frames, 2, num_anythings)
        The second dimension (2) represents (x, y) pixel coordinates.
        """
        if self._topoManFilled is None:
            self._topoManFilled, self._F_map, self._sampson_err = TopoUtils.fill_pix_via_F_w_log(self.topoManAllFrames)
        return self._topoManFilled

    @overload
    def anything_id_index(self, id_x_: int) ->  str: ...
    @overload
    def anything_id_index(self, id_x_: str) -> int: ...
    def anything_id_index(self, id_x_: int | str) -> str | int:
        """
        Converts between an AnyThing ID (string) and its corresponding integer index, or vice versa.

        If a string ID is provided, the method returns its integer index.
        If an integer index is provided, it returns the corresponding string ID.

        Args:
            id_x_ (int | str): The AnyThing identifier — either the string ID or its integer index.

        Returns:
            int | str: The corresponding index (if a string ID was given) or the corresponding ID (if an index was given).

        Raises:
            ValueError: If `id_x_` is neither a string nor an integer, or if the provided ID/index does not exist.
        """
        if isinstance(id_x_, str):
            index = self._anything_id_to_index_map.get(id_x_, -1)
            if index == -1:
                raise ValueError()
            return index
        elif isinstance(id_x_, (int, np.integer)) and not isinstance(id_x_, bool):
            id = self._anything_id_to_index_map.inv.get(id_x_, "")
            if id == "":
                raise ValueError()
            return id
        else:
            raise ValueError(f"id_x_ :: {id_x_} :: should either be str or int.")

    @overload
    def frame_id_index(self, id_x_: int) -> str: ...
    @overload
    def frame_id_index(self, id_x_: str) -> int: ...
    def frame_id_index(self, id_x_: int | str) ->  str | int:
        """
        Converts between a frame ID (string) and its corresponding integer index, or vice versa.

        If a string ID is provided, the method returns its integer index.
        If an integer index is provided, it returns the corresponding string ID.

        Args:
            id_x_ (int | str): The frame identifier — either the string ID or its integer index.

        Returns:
            int | str: The corresponding index (if a string ID was given) or the corresponding ID (if an index was given).

        Raises:
            ValueError: If `id_x_` is neither a string nor an integer, or if the provided ID/index does not exist.
        """
        if isinstance(id_x_, str):
            index = self._frame_id_to_index_map.get(id_x_, -1)
            if index == -1:
                raise ValueError(f"Frame with id {id_x_} is not found.")
            return index
        elif isinstance(id_x_, (int, np.integer)) and not isinstance(id_x_, bool):
            id = self._frame_id_to_index_map.inv.get(id_x_, "")
            if id == "":
                raise ValueError(f"Frame at index {id_x_} is not found.")
            return id
        else:
            raise ValueError(f"id_x_ :: {id_x_} :: should either be str or int.")

    def _add_anything_w_id(self, anything_id: str) -> None:
        """
        Adds a new AnyThing to the environment if it doesn't already exist.
        Updates the internal ID-to-index mapping and invalidates relevant caches.

        Args:
            anything_id (str): The ID of the AnyThing to add.
        """
        if anything_id not in self.anything_ids:
            new_index = self.num_anythings
            self._anything_ids_added.append(anything_id)
            self._anything_id_to_index_map[anything_id] = new_index
            self.invalidateTopoManCache()

    def add_data(self,
                 *,
                 new_frame_id: Optional[str] = None,
                 new_anything_ids: Optional[List[str]] = None) -> None:
        """
        Adds new data (AnyThings and/or frames) of the topological environment.
        This method handles updating internal mapping and invalidating caches as necessary.

        Args:
        new_frame_id (Optional[str]): An optional frame ID to register.
            The frame is added only if it is not already registered and at least one known AnyThing contains an observation in it.
        new_anything_ids (Optional[List[str]]): An optional list of new AnyThing IDs to add. If None, no AnyThings are added.
        """
        # Add anythings if any
        if new_anything_ids is not None:
            for new_anything_id in new_anything_ids:
                self._add_anything_w_id(new_anything_id)

        # Add frame if any
        if new_frame_id is None or new_frame_id in self.frame_ids:
            return

        anything_found_in_frame = False
        for known_anything_id in self.anything_ids:
            anything = self.anything_manager.get(known_anything_id, optional=False)
            assert anything is not None
            if new_frame_id in anything.frame_ids:
                self.invalidateTopoManCache()
                anything_found_in_frame = True
                break

        if anything_found_in_frame:
            new_frame_index = self.num_frames
            self._frame_ids_added.append(new_frame_id)
            self._frame_id_to_index_map[new_frame_id] = new_frame_index

    def createTopoManPix(self,
                         *,
                         frame_type: str,
                         anything_type: str) -> np.ndarray:
        """
        Calculates a 3D topological pixel map for a specific subset of frames and
        AnyThings. The output array is globally indexed, meaning its dimensions correspond
        to the total number of frames and AnyThings in the environment, with non-relevant
        entries filled with NaN.

        Args:
            frame_type (str): Specifies which frames to include ("orig", "added" or "all").
            anything_type (str): Specifies which AnyThings to include ("orig", "added" or "all").

        Returns:
            np.ndarray: A 3D numpy array of shape (num_frames, 2, num_anythings) containing
                        pixel coordinates. Entries not relevant to the specified types are NaN.
                        The second dimension (2) represents (x, y) pixel coordinates.

        Raises:
            ValueError: If an invalid 'frame_type' or 'anything_type' is provided.
        """
        if frame_type == "orig":
            frame_ids = self.frame_ids_orig
        elif frame_type == "added":
            frame_ids = self.frame_ids_added
        elif frame_type == "all":
            frame_ids = self.frame_ids
        else:
            raise ValueError(f"frame_type :: {frame_type} :: can only be 'all', 'orig' or 'added'.")

        if anything_type == "orig":
            anything_ids = self.anything_ids_orig
        elif anything_type == "added":
            anything_ids = self.anything_ids_added
        elif anything_type == "all":
            anything_ids = self.anything_ids
        else:
            raise ValueError(f"anything_type :: {anything_type} :: can only be 'all', 'orig' or 'added'.")

        topoManPix: np.ndarray = np.full(
            (
                self.num_frames, 2, self.num_anythings
            ), np.nan, dtype=np.float64
        )

        if not frame_ids or not anything_ids:
            return topoManPix

        logger.info("---- Topological Manifold is being created...")
        for anything_id in tqdm(anything_ids, desc="Going over each AnyThing inside TopoEnv..."):
            anything = self.anything_manager.get(anything_id, optional=False)
            assert anything is not None
            topoManPix[
                np.array([self.frame_id_index(frame_id) for frame_id in frame_ids], dtype=int),
                :,
                self.anything_id_index(anything_id)
            ] = np.array(anything.center_in_frames(frame_ids)).reshape(len(frame_ids), 2)

        return topoManPix

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "video_path": self.videoContainer.path,
            "anything_ids_orig": self.anything_ids_orig,
            "anything_ids_added": self.anything_ids_added,
            "frame_ids_added": self.frame_ids_added,
            "preprocessing_params": self.preprocessing_params,
            "max_dim_for_processing": self._max_dim_for_processing,
            "cache_dir": self.cache_dir
        }

    @classmethod
    def from_dict(cls,
                  data: Dict[str, Any]):

        anything_ids = data["anything_ids_orig"] if data["anything_ids_orig"] else []
        video_path = data["video_path"]
        id = data["id"]
        preprocessing_params = data["preprocessing_params"]
        max_dim_for_processing = data["max_dim_for_processing"]
        cache_dir = data["cache_dir"]
        topoenv = cls(
            cache_dir=cache_dir,
            anything_ids=anything_ids,
            video_path=video_path,
            max_dim_for_processing=max_dim_for_processing,
            preprocessing_params=preprocessing_params,
            id=id,
            is_dirty=False
        )

        topoenv.add_data(new_anything_ids=data.get("anything_ids_added"))

        for frame_id in data.get("frame_ids_added", []):
            topoenv.add_data(new_frame_id=frame_id)

        topoenv.is_dirty = False
        return topoenv


class TopoEnvCacheManager(BaseCacheManager):
    """A cache manager for TopoEnv objects."""
    def __init__(self,
                 cache_dir: str,
                 use_cache_disk: bool = True):
        super().__init__(cache_dir, TopoEnv, use_cache_disk=use_cache_disk)

    def _get_filepath(self, obj_id: str) -> str:
        return os.path.join(self.cache_dir, f"{obj_id}.pkl")

    def _load_from_file(self, filepath: str) -> TopoEnv:
        with open(filepath, "rb") as f:
            data = pickle.load(f)
        return TopoEnv.from_dict(data)


class TopoUtils:
    """
    A collection of static utility methods for geometric computations,
    including Essential Matrix estimation, robust epipolar-line intersection,
    and reconstruction of missing image coordinates using epipolar geometry.
    """
    @staticmethod
    def solve_point_from_lines(lines_3xN: np.ndarray,
                               *,
                               line_weights: Optional[np.ndarray] = None,
                               theta_min_degree: float = 12.0,
                               bin_count: int = 10,
                               max_per_bin: int = 12,
                               huber_delta_pts: float = 2.0,
                               huber_delta_lines: float = 1.0,
                               refine_iters: int = 10,
                               tau_dist: float = 2.0,
                               roi: Optional[Tuple[float, float, float, float]] = None,
                               rng: Optional[np.random.Generator] = None
                               ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Hybrid epipolar-line intersection:
        1) Normalize lines.
        2) Thin lines using angular bins.
        3) Intersect pairs with sufficient angular separation; weight intersections by w_i*w_j*sin(angle).
        4) Robustly aggregate intersections using a weighted Huber mean to obtain the initial estimate p0.
        5) Identify an initial consensus set among all valid normalized lines using |distance| < tau_dist.
        6) Refine the intersection using Huber IRLS on the consensus lines.
        7) Recompute the final inlier mask using the refined intersection.

        Args:
            lines_3xN: (3,N) lines l=[a,b,c] s.t. a*u + b*v + c = 0 (same unit system as thresholds).
            line_weights: optional per-line confidence weights of shape (N,).
                Weights must be finite and nonnegative, with at least one
                positive weight among the geometrically valid lines.
                Higher weights indicate greater confidence.
            theta_min_degree: minimum angle between line normals to accept a pair (degrees).
            bin_count: number of angle bins for thinning in [0, π).
            max_per_bin: keep at most this many lines per bin.
            huber_delta_pts: Dimensionless Huber threshold for standardized
                intersection-point residuals.
            huber_delta_lines: Dimensionless Huber threshold for standardized
                line-distance residuals.
            refine_iters: IRLS iterations.
            tau_dist: inlier distance threshold to the point (same units as lines).
            roi: optional (xmin, ymin, xmax, ymax) to reject far-away vanishing-point solutions.
            rng: NOT USED, optional numpy Generator.

        Returns:
            (p, inlier_mask):
            p: (2,) final intersection (np.nan, np.nan on failure).
            inlier_mask: (N,) bool over the ORIGINAL input lines.
        """
        if rng is None:
            rng = np.random.default_rng()

        if lines_3xN.size == 0:
            N = lines_3xN.shape[1] if lines_3xN.ndim == 2 else 0
            return np.array([np.nan, np.nan], dtype=np.float64), np.zeros(N, dtype=bool)

        # ---- 1) Normalize lines (||[a,b]|| = 1); track original indices
        L = lines_3xN.T.astype(np.float64, copy=True)  # (N,3)
        N = L.shape[0]
        ab = L[:, :2]
        nrm = np.linalg.norm(ab, axis=1, keepdims=True)
        ok = (nrm[:, 0] > 0) & np.isfinite(L).all(axis=1)
        if not np.any(ok):
            return np.array([np.nan, np.nan], dtype=np.float64), np.zeros(N, dtype=bool)

        idx_keep = np.where(ok)[0]

        L = L[idx_keep]
        nrm = nrm[idx_keep]

        L[:, :2] /= nrm
        L[:, 2]  /= nrm[:, 0]
        M = L.shape[0]

        # Per-line weights (default 1)
        if line_weights is not None:
            weights = np.asarray(line_weights, dtype=np.float64)

            if weights.shape != (N,):
                raise ValueError("line_weights must have shape (N,).")

            if not np.all(np.isfinite(weights)) or np.any(weights < 0):
                raise ValueError("line weights must be finite and nonnegative.")

            w_all = weights[idx_keep]

            if not np.any(w_all > 0):
                raise ValueError("At least one valid line must have a positive weight.")

            w_all = np.clip(w_all, 1e-6, None)
            w_all = w_all / np.mean(w_all)

        else:
            w_all = np.ones(M, dtype=np.float64)

        # ---- 2) Angle-bin thinning (keeps runtime tame when many lines are parallel)
        theta = np.mod(np.arctan2(L[:,1], L[:,0]), np.pi)  # [0, π)
        bin_edges = np.linspace(0.0, np.pi, bin_count + 1)
        bins = np.minimum(np.digitize(theta, bin_edges) - 1, bin_count - 1)

        keep_mask = np.zeros(M, dtype=bool)
        for b in range(bin_count):
            idx_b = np.where(bins == b)[0]
            if idx_b.size == 0:
                continue
            if idx_b.size > max_per_bin:
                sel = idx_b[np.argsort(w_all[idx_b])[-max_per_bin:]]
            else:
                sel = idx_b
            keep_mask[sel] = True

        L_thin = L[keep_mask]
        w_thin = w_all[keep_mask]
        idx_map = idx_keep[keep_mask]  # map back to original indices
        Mt = L_thin.shape[0]
        if Mt < 2:
            return np.array([np.nan, np.nan], dtype=np.float64), np.zeros(N, dtype=bool)

        # ---- 3) Intersect valid pairs (angle ≥ theta_min_degree), weight by w_i*w_j*sin(angle)
        iu, ju = np.triu_indices(Mt, k=1)
        n1 = L_thin[iu, :2]; n2 = L_thin[ju, :2]
        dot = np.clip(np.sum(n1*n2, axis=1), -1.0, 1.0)
        alpha = np.arccos(np.abs(dot))
        theta_min = np.deg2rad(theta_min_degree)
        valid = alpha >= theta_min
        if not np.any(valid):
            return np.array([np.nan, np.nan], dtype=np.float64), np.zeros(N, dtype=bool)

        iu = iu[valid]; ju = ju[valid]; alpha = alpha[valid]
        l1 = L_thin[iu]; l2 = L_thin[ju]
        h = np.cross(l1, l2)  # (K,3)
        finite = np.isfinite(h).all(axis=1) & (np.abs(h[:,2]) > 1e-12)
        if not np.any(finite):
            return np.array([np.nan, np.nan], dtype=np.float64), np.zeros(N, dtype=bool)

        h = h[finite]; alpha = alpha[finite]; iu = iu[finite]; ju = ju[finite]
        P = (h[:, :2].T / h[:, 2]).T  # (K,2)
        wp = (w_thin[iu] * w_thin[ju]) * np.sin(alpha)
        wp = np.clip(wp, 1e-12, None)

        if roi is not None:
            xmin, ymin, xmax, ymax = roi
            in_roi = (P[:,0] >= xmin-50) & (P[:,0] <= xmax+50) & (P[:,1] >= ymin-50) & (P[:,1] <= ymax+50)
            if not np.any(in_roi):
                return np.array([np.nan, np.nan], dtype=np.float64), np.zeros(N, dtype=bool)
            P = P[in_roi]; wp = wp[in_roi]
        if P.shape[0] == 0:
            return np.array([np.nan, np.nan], dtype=np.float64), np.zeros(N, dtype=bool)

        # ---- 4) Weighted Huber mean (intersections)
        def _weighted_huber_mean(Pxy: np.ndarray, W: np.ndarray, delta: float, iters: int) -> np.ndarray:
            p = (W[:,None] * Pxy).sum(axis=0) / (W.sum() + 1e-12)
            for _ in range(iters):
                r = Pxy - p
                d = np.linalg.norm(r, axis=1) + 1e-12
                med = np.median(d)
                sigma = 1.4826 * np.median(np.abs(d - med))
                if not np.isfinite(sigma) or sigma < 1e-9:
                    sigma = 1.0
                t = d / sigma
                hub = np.ones_like(t)
                big = t > delta
                hub[big] = (delta / (t[big] + 1e-12))
                W_eff = W * hub
                p_new = (W_eff[:,None] * Pxy).sum(axis=0) / (W_eff.sum() + 1e-12)
                if not np.all(np.isfinite(p_new)) or np.linalg.norm(p_new - p) < 1e-6:
                    return p_new
                p = p_new
            return p

        p0 = _weighted_huber_mean(P, wp, huber_delta_pts, max(3, refine_iters // 2))
        if not np.all(np.isfinite(p0)):
            return np.array([np.nan, np.nan], dtype=np.float64), np.zeros(N, dtype=bool)

        # ---- 5) Consensus over ALL normalized lines
        d_all = np.abs(L[:,0]*p0[0] + L[:,1]*p0[1] + L[:,2])
        inliers_L = d_all < tau_dist
        if np.count_nonzero(inliers_L) < 2:
            d_thin = np.abs(L_thin[:,0]*p0[0] + L_thin[:,1]*p0[1] + L_thin[:,2])
            inliers_thin = d_thin < tau_dist
            if np.count_nonzero(inliers_thin) < 2:
                return np.array([np.nan, np.nan], dtype=np.float64), np.zeros(N, dtype=bool)
            L_ref = L_thin[inliers_thin]; w_ref = w_thin[inliers_thin]
            inlier_mask_orig = np.zeros(N, dtype=bool)
            inlier_mask_orig[idx_map[inliers_thin]] = True
        else:
            L_ref = L[inliers_L]; w_ref = w_all[inliers_L]
            inlier_mask_orig = np.zeros(N, dtype=bool)
            inlier_mask_orig[idx_keep[inliers_L]] = True

        # ---- 6) Final Huber IRLS on line distances
        def _irls_huber_lines(Lrows: np.ndarray, p_init: np.ndarray,
                            w0: np.ndarray | None, delta: float, iters: int) -> np.ndarray:
            p = p_init.copy()
            w_base = np.ones(Lrows.shape[0]) if w0 is None else np.clip(w0, 1e-8, None)
            for _ in range(iters):
                r = Lrows[:,0]*p[0] + Lrows[:,1]*p[1] + Lrows[:,2]
                med = np.median(r)
                sigma = 1.4826 * np.median(np.abs(r - med))
                if not np.isfinite(sigma) or sigma < 1e-9:
                    sigma = 1.0
                t = r / sigma
                hub = np.ones_like(t)
                big = np.abs(t) > delta
                hub[big] = delta / (np.abs(t[big]) + 1e-12)
                w = w_base * hub
                sqrt_w = np.sqrt(w)
                A = sqrt_w[:,None] * Lrows[:, :2]
                b = -sqrt_w * Lrows[:, 2]
                ATA = A.T @ A + 1e-10 * np.eye(2)
                ATb = A.T @ b
                try:
                    p_new = np.linalg.solve(ATA, ATb)
                except np.linalg.LinAlgError:
                    p_new, *_ = np.linalg.lstsq(A, b, rcond=None)
                if not np.all(np.isfinite(p_new)) or np.linalg.norm(p_new - p) < 1e-6:
                    return p_new
                p = p_new
            return p

        p_final = _irls_huber_lines(L_ref, p0, w_ref, huber_delta_lines, refine_iters)
        if not np.all(np.isfinite(p_final)):
            return np.array([np.nan, np.nan], dtype=np.float64), np.zeros(N, dtype=bool)

        # ---- 7) Final consensus over all valid normalized lines
        d_all_final = np.abs(
            L[:, 0] * p_final[0]
            + L[:, 1] * p_final[1]
            + L[:, 2]
        )

        # Restore the original input indexing.
        inlier_mask_out = np.zeros(N, dtype=bool)
        inlier_mask_out[idx_keep] = d_all_final < tau_dist

        if roi is not None:
            xmin, ymin, xmax, ymax = roi
            if not (xmin-50 <= p_final[0] <= xmax+50 and ymin-50 <= p_final[1] <= ymax+50):
                return np.array([np.nan, np.nan], dtype=np.float64), np.zeros(N, dtype=bool)

        return p_final.astype(np.float64), inlier_mask_out

    @staticmethod
    def sampson_error(E: np.ndarray, pts1: np.ndarray, pts2: np.ndarray) -> np.ndarray:
        """
        Computes the squared Sampson error for each correspondence.

        Args:
            E (np.ndarray): (3, 3) Fundamental or Essential matrix.
            pts1 (np.ndarray): (2, N) points in the first image.
            pts2 (np.ndarray): (2, N) corresponding points in the second image.

        Returns:
            np.ndarray: (N,) squared Sampson errors.

        Notes:
            Points should be expressed in pixel coordinates when E is a
            Fundamental matrix, or normalized camera coordinates when E
            is an Essential matrix.
        """
        ones = np.ones((1, pts1.shape[1]))
        x1 = np.vstack([pts1, ones])
        x2 = np.vstack([pts2, ones])

        Ex1 = E @ x1
        Etx2 = E.T @ x2
        num = np.sum(x2 * Ex1, axis=0) ** 2
        denom = Ex1[0]**2 + Ex1[1]**2 + Etx2[0]**2 + Etx2[1]**2
        denom = np.maximum(denom, 1e-15)
        return num / denom  # squared Sampson error per correspondence

    @staticmethod
    def _normalize_points(pts: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Hartley-like similarity normalization (centroid -> 0, mean dist -> sqrt(2))"""
        mu = np.mean(pts, axis=1, keepdims=True)            # (2,1)
        centered = pts - mu                                 # (2,N)
        d = np.sqrt(np.sum(centered**2, axis=0))            # (N,)
        mean_d = np.mean(d)
        s = (np.sqrt(2.0) / mean_d) if mean_d > 0 else 1.0
        T = np.array([[s, 0, -s * mu[0, 0]],
                        [0, s, -s * mu[1, 0]],
                        [0, 0, 1]], dtype=float)
        pts_h = np.vstack([pts, np.ones((1, pts.shape[1]), dtype=float)])  # (3,N)
        pts_n = T @ pts_h                                                  # (3,N)
        return pts_n[:2, :], T

    @staticmethod
    def eight_point_normalized_fundamental(pts1: np.ndarray, pts2: np.ndarray) -> Optional[np.ndarray]:
        """
        Estimates the Fundamental Matrix (F) between two views using the normalized
        8-point algorithm.

        Args:
            pts1 (np.ndarray): (2, N) corresponding points in image 1 (finite).
            pts2 (np.ndarray): (2, N) corresponding points in image 2 (finite).
        Returns:
            Optional[np.ndarray]: (3, 3) Fundamental matrix, or None
                if the mean Sampson distance exceeds 5 pixels.
        """
        # ---- Input checks
        if pts1.ndim != 2 or pts2.ndim != 2 or pts1.shape[0] != 2 or pts2.shape[0] != 2:
            raise ValueError("pts1 and pts2 must be shaped (2, N)")
        if pts1.shape[1] != pts2.shape[1]:
            raise ValueError("pts1 and pts2 must have the same number of correspondences")
        N = pts1.shape[1]
        if N < 8:
            raise ValueError("At least 8 correspondences are required.")
        if not np.all(np.isfinite(pts1)) or not np.all(np.isfinite(pts2)):
            raise ValueError("All input points must be finite (no NaNs/Infs).")

        pts1 = pts1.astype(float, copy=False)
        pts2 = pts2.astype(float, copy=False)

        p1n, T1 = TopoUtils._normalize_points(pts1)
        p2n, T2 = TopoUtils._normalize_points(pts2)

        x1, y1 = p1n[0, :], p1n[1, :]
        x2, y2 = p2n[0, :], p2n[1, :]

        # ---- Build A (Nx9)
        A = np.stack([
            x2 * x1,  x2 * y1,  x2,
            y2 * x1,  y2 * y1,  y2,
            x1,       y1,       np.ones(N, dtype=float)
        ], axis=1)

        # ---- Solve A e = 0 via SVD
        U, S, Vt = np.linalg.svd(A, full_matrices=(N==8))

        f = Vt[-1, :]
        F_hat = f.reshape(3, 3)

        # ---- Enforce rank-2
        Ue, Se, Vte = np.linalg.svd(F_hat)
        Se[-1] = 0
        F_hat = Ue @ np.diag(Se) @ Vte

        # ---- Denormalize
        F = T2.T @ F_hat @ T1

        # ---- Normalize scale for determinism
        nrm = np.linalg.norm(F)
        if nrm > 0:
            F = F / nrm

        average_sampson_err = np.mean(
            np.sqrt(
                TopoUtils.sampson_error(F, pts1, pts2)
                )
            )

        if average_sampson_err > 5:
            logger.debug(f"Average Sampson Error of {average_sampson_err} is achieved. Invalid F.")
            return None
        else:
            logger.debug(f"Average Sampson Error of {average_sampson_err} is achieved. Valid F.")

        return F

    @staticmethod
    def eight_point_normalized_essential(pts1: np.ndarray, pts2: np.ndarray) -> Optional[np.ndarray]:
        """
        Estimates the Essential Matrix (E) between two views using the normalized
        8-point algorithm.

        Args:
            pts1 (np.ndarray): (2, N) corresponding points in image 1,
                expressed in normalized camera coordinates.
            pts2 (np.ndarray): (2, N) corresponding points in image 2,
                expressed in normalized camera coordinates.

        Returns:
            Optional[np.ndarray]: (3, 3) Frobenius-normalized
                Essential matrix, or None if the mean Sampson distance
                exceeds 0.1 in normalized camera coordinates.
        """
        # ---- Input checks
        if pts1.ndim != 2 or pts2.ndim != 2 or pts1.shape[0] != 2 or pts2.shape[0] != 2:
            raise ValueError("pts1 and pts2 must be shaped (2, N).")
        if pts1.shape[1] != pts2.shape[1]:
            raise ValueError("pts1 and pts2 must have the same number of correspondences.")
        N = pts1.shape[1]
        if N < 8:
            raise ValueError("At least 8 correspondences are required.")
        if not np.all(np.isfinite(pts1)) or not np.all(np.isfinite(pts2)):
            raise ValueError("All input points must be finite (no NaNs/Infs).")

        pts1 = pts1.astype(float, copy=False)
        pts2 = pts2.astype(float, copy=False)

        p1n, T1 = TopoUtils._normalize_points(pts1)
        p2n, T2 = TopoUtils._normalize_points(pts2)

        x1, y1 = p1n[0, :], p1n[1, :]
        x2, y2 = p2n[0, :], p2n[1, :]

        # ---- Build A (N×9)
        A = np.stack([
            x2 * x1,  x2 * y1,  x2,
            y2 * x1,  y2 * y1,  y2,
            x1,       y1,       np.ones(N, dtype=float)
        ], axis=1)

        # ---- Solve A e = 0 via SVD
        U, S, Vt = np.linalg.svd(A, full_matrices=(N==8))

        e = Vt[-1, :]
        E_hat = e.reshape(3, 3)

        # ---- Denormalize
        E = T2.T @ E_hat @ T1

        # ---- Enforce rank-2 with equal top two singular values
        Ue, Se, Vte = np.linalg.svd(E)
        s = 0.5 * (Se[0] + Se[1])
        E = Ue @ np.diag([s, s, 0.0]) @ Vte

        # ---- Normalize scale for determinism
        nrm = np.linalg.norm(E)
        if nrm > 0:
            E = E / nrm

        average_sampson_err = np.mean(
            np.sqrt(
                TopoUtils.sampson_error(E, pts1, pts2)
                )
            )

        if average_sampson_err > 1e-1:
            logger.warning(f"Sampson Error of {average_sampson_err} is achieved. Invalid E.")
            return None

        return E

    @staticmethod
    def eight_point_normalized_RANSAC(pts1: np.ndarray,
                                      pts2: np.ndarray,
                                      calibrated: bool =False,
                                      *,
                                      n_iter: int = 500,
                                      thresh: float = 5,
                                      min_samples: int = 8,
                                      min_inliers: int = 12,
                                      rng: Optional[np.random.Generator] = None
                                      ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Estimates either the Fundamental Matrix (uncalibrated) or the Essential Matrix (calibrated)
        using the normalized 8-point algorithm within a RANSAC loop.

        The method automatically switches between the two formulations based on the 'calibrated' flag.

        If `calibrated=True`, the algorithm estimates the **Essential matrix (E)**:
        - Input points (`pts1`, `pts2`) must be in **normalized camera coordinates**
          (i.e., already corrected for camera intrinsics).
        - The inlier `thresh` is applied to the Sampson error computed in these normalized units.

        If `calibrated=False`, the algorithm estimates the **Fundamental matrix (F)**:
        - Input points (`pts1`, `pts2`) should be in **pixel coordinates**.
        - The inlier `thresh` is applied to the Sampson error computed in pixel units.

        Args:
            pts1 (np.ndarray): (2, N) array of corresponding points in image 1.
            pts2 (np.ndarray): (2, N) array of corresponding points in image 2.
            calibrated (bool): If True, estimate Essential matrix from normalized coordinates;
                               otherwise, estimate Fundamental matrix from pixel coordinates.
            n_iter (int): Maximum number of RANSAC iterations.
            thresh (float): Inlier threshold on the squared Sampson error,
                expressed in squared pixel units for Fundamental matrix
                estimation or squared normalized-coordinate units for
                Essential matrix estimation.
            min_samples (int): Number of correspondences sampled per RANSAC iteration (default 8).
                Must be at least 8.
            min_inliers (int): Minimum number of inliers required to accept a model.
            rng (np.random.Generator, optional): Random number generator for reproducibility.

        Returns:
            Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
                - M_best (np.ndarray): The estimated 3x3 matrix (E if calibrated, else F).
                - inlier_mask (np.ndarray): Boolean array of shape (N,) indicating inlier correspondences.
                Returns (None, None) if no valid model is found.

        Notes:
            - The algorithm uses the normalized 8-point estimator (`eight_point_normalized_essential` or
                `eight_point_normalized_fundamental`) as the minimal solver.
            - Sampson error is used as the geometric error measure in RANSAC scoring.
            - The returned matrix is Frobenius-normalized for numerical stability.
            - The default threshold of 5 is intended for pixel-coordinate
                estimation. When calibrated=True, the caller should provide
                a threshold appropriate for normalized camera coordinates.
        """
        # ---- Checks
        if pts1.ndim != 2 or pts2.ndim != 2 or pts1.shape[0] != 2 or pts2.shape[0] != 2:
            raise ValueError("pts1 and pts2 must be shape (2,N).")
        if pts1.shape[1] != pts2.shape[1]:
            raise ValueError("pts1 and pts2 must have the same number of correspondences.")
        if not np.all(np.isfinite(pts1)) or not np.all(np.isfinite(pts2)):
            raise ValueError("All input points must be finite (no NaN/Infs).")
        if min_samples < 8:
            raise ValueError(f"min_samples should be at least 8 but it is {min_samples}.")

        N = pts1.shape[1]
        if N < max(min_samples, min_inliers):
            return None, None

        if rng is None:
            rng = np.random.default_rng()
        assert rng is not None

        best_E: Optional[np.ndarray] = None
        best_inliers: Optional[np.ndarray] = None
        best_count = 0
        best_mean_error = np.inf

        idx_all = np.arange(N)

        cap_comb = math.comb(N, min_samples) if N < 50 else math.inf
        n_iter = int(min(n_iter, cap_comb if cap_comb > 0 else n_iter))

        if calibrated:
            eight_point_estimator = TopoUtils.eight_point_normalized_essential
        else:
            eight_point_estimator = TopoUtils.eight_point_normalized_fundamental

        for _ in range(n_iter):
            # 1) Minimal sample
            sample = rng.choice(idx_all, size=min_samples, replace=False)
            E_min = eight_point_estimator(pts1[:, sample], pts2[:, sample])
            if E_min is None or not np.all(np.isfinite(E_min)):
                continue

            # 2) Score by Sampson error (guard against zeros/NaNs)
            errs = TopoUtils.sampson_error(E_min, pts1, pts2)
            errs = np.where(np.isfinite(errs), errs, np.inf)
            inliers = errs < thresh
            count = int(np.count_nonzero(inliers))
            if count < max(min_inliers, min_samples):
                continue

            E_candidate = E_min
            inliers_candidate = inliers
            count_candidate = count
            mean_candidate = float(np.mean(errs[inliers]))

            # 3) Re-fit on inliers
            E_refit = eight_point_estimator(pts1[:, inliers], pts2[:, inliers])

            if E_refit is not None and np.all(np.isfinite(E_refit)):

                # 4) Re-score the refitted model
                errs_refit = TopoUtils.sampson_error(E_refit, pts1, pts2)
                errs_refit = np.where(np.isfinite(errs_refit), errs_refit, np.inf)

                inliers_refit = errs_refit < thresh
                count_refit = int(np.count_nonzero(inliers_refit))

                if count_refit > 0:
                    mean_refit = float(np.mean(errs_refit[inliers_refit]))

                    # Accept refitting only if it improves the RANSAC score
                    if (
                            count_refit > count_candidate
                        ) or (
                            count_refit == count_candidate and mean_refit < mean_candidate
                        ):
                        E_candidate = E_refit
                        inliers_candidate = inliers_refit
                        count_candidate = count_refit
                        mean_candidate = mean_refit

            if count_candidate > best_count or (
                best_E is not None and
                count_candidate == best_count and
                mean_candidate < best_mean_error
            ):
                best_count = count_candidate
                best_E = E_candidate
                best_inliers = inliers_candidate
                best_mean_error = mean_candidate

            if best_count == N:
                break

        if best_E is None or best_count < max(min_inliers, min_samples):
            return None, None

        return best_E, best_inliers

    @staticmethod
    def fill_single_frame_via_F(
        memory: np.ndarray,
        proposal: np.ndarray,
        *,
        min_common: int = 8,
        min_refs: int = 2
    ) -> Tuple[np.ndarray, Dict[int, np.ndarray]]:
        """
        Fill missing pixel coordinates in a single proposal frame using epipolar
        geometry with respect to a set of memory frames.

        Args:
            memory (np.ndarray): (F_mem, 2, A) array of (x, y) coordinates.
                                 This is the "map" / memory (can contain NaNs).
            proposal (np.ndarray): (2, A) or (1, 2, A) array for a single frame.
                                   NaNs indicate missing objects to be estimated.
            min_common (int): Minimum shared visible objects between a memory
                              frame and the proposal to estimate a Fundamental
                              matrix F_iP.
            min_refs (int): Minimum distinct epipolar lines required to solve a
                            missing point via line intersection.

        Returns:
            proposal_filled (np.ndarray): (2, A) proposal with some NaNs filled.
            Fmap (dict): {i: F_iP} Fundamental matrices from memory frame i to
                         the proposal frame, x_proposal^T F_iP x_memory = 0.
        """
        assert memory.ndim == 3 and memory.shape[1] == 2, \
            "memory must be (F_mem, 2, A)"
        A = memory.shape[2]

        debug=False

        # Accept (2, A) or (1, 2, A) for proposal
        if proposal.ndim == 3:
            assert proposal.shape[0] == 1 and proposal.shape[1] == 2, \
                "proposal must be (2, A) or (1, 2, A)"
            proposal_2A = proposal[0]
        else:
            assert proposal.ndim == 2 and proposal.shape[0] == 2, \
                "proposal must be (2, A) or (1, 2, A)"
            proposal_2A = proposal

        assert proposal_2A.shape[1] == A, \
            "memory and proposal must have the same number of objects (A)."

        F_mem, _, _ = memory.shape

        proposal_filled = proposal_2A.copy()

        # -------------------------------------------------------------
        # 1) Estimate F_iP between each memory frame i and the proposal
        # -------------------------------------------------------------
        Fmap: Dict[int, np.ndarray] = {}

        for i in range(F_mem):
            Xi = memory[i]  # (2, A)
            # visible in both memory frame i and proposal
            vi = np.all(np.isfinite(Xi), axis=0)
            vp = np.all(np.isfinite(proposal_2A), axis=0)
            common = vi & vp
            n_common = int(np.count_nonzero(common))
            if n_common < max(8, min_common):
                continue

            Xi_c = Xi[:, common]              # (2, N_common)
            Xp_c = proposal_2A[:, common]     # (2, N_common)

            # 8-point normalized RANSAC in pixel domain
            F_iP, _ = TopoUtils.eight_point_normalized_RANSAC(
                Xi_c,
                Xp_c,
                calibrated=False,
                min_samples=8,
                min_inliers=8,
                thresh=5
            )
            if F_iP is None:
                # if debug:
                #     print("F_iP returned None")
                continue

            Fmap[i] = F_iP

        if len(Fmap) == 0:
            # Nothing we can do; return proposal as-is
            if debug:
                print(f"{len(Fmap)=} == 0")
            return proposal_filled, Fmap

        # -------------------------------------------------------------
        # 2) For each object a, if missing in proposal, intersect lines
        #    coming from memory frames that see this object.
        # -------------------------------------------------------------
        for a in range(A):
            # If proposal already has this object's coordinates, skip
            if np.all(np.isfinite(proposal_2A[:, a])):
                continue

            lines = []
            for i, F_iP in Fmap.items():
                Xi = memory[i, :, a]  # (2,)
                if not np.all(np.isfinite(Xi)):
                    if debug:
                        print("not np.all(np.isfinite(Xi))")
                    continue

                # homogeneous memory point
                xih = np.array([Xi[0], Xi[1], 1.0], dtype=np.float64)
                # epipolar line in proposal frame: x_p^T F_iP x_i = 0
                l_p = F_iP @ xih  # (3,)
                if np.allclose(l_p[:2], 0):
                    if debug:
                        print("np.allclose(l_p[:2], 0)")
                    continue

                lines.append(l_p.reshape(3, 1))

            if len(lines) < min_refs:
                # not enough independent epipolar lines to solve
                if debug:
                    print(f"{len(lines)=} < min_refs")
                continue

            L = np.hstack(lines)  # (3, N_lines)

            # Reuse the existing line intersection solver
            p_hat, _ = TopoUtils.solve_point_from_lines(L)  # (2,) + diagnostics
            if np.all(np.isfinite(p_hat)):
                if debug:
                    print("np.all(np.isfinite(p_hat))")
                proposal_filled[:, a] = p_hat.astype(np.float64)

        return proposal_filled, Fmap

    @staticmethod
    def fill_pix_via_F_w_log(
        P: np.ndarray, *, min_common: int = 12, min_refs: int = 2
    ) -> Tuple[np.ndarray, Dict[Tuple[int, int], np.ndarray], Dict[Tuple[int, int], np.ndarray]]:
        """
        Fills NaN pixel coordinates using Fundamental matrices and epipolar geometry.
        For planar scenes or pure rotation, you may add a Homography fallback separately.

        Args:
            P (np.ndarray): (F, 2, A) array of (x, y) undistorted pixel coordinates.
            min_common (int): Minimum shared points to estimate a Fundamental matrix F_ij.
            min_refs (int): Minimum distinct epipolar lines required to solve a missing point.

        Returns:
            P_out (np.ndarray): Filled (F, 2, A).
            Fmap (dict): {(i, j): F_ij} where x_j^T F_ij x_i = 0, i < j.
            F_sampson (dict): {(i, j): errors}, where errors is an (N_common,) array of
                squared Sampson errors for correspondences observed in both frames. The errors
                follow the original object-index order, restricted to the common correspondences.
        """
        assert P.ndim == 3 and P.shape[1] == 2, "P must be (F, 2, A)"
        Fnum, _, A = P.shape
        P_out = P.copy()

        # ---- 1) Estimate F between frame pairs
        Fmap: Dict[Tuple[int, int], np.ndarray] = {}
        F_sampson: Dict[Tuple[int, int], np.ndarray] = {}

        logger.info(f"---- Filling the missing data inside the Topological Matrix.")

        for i in tqdm(range(Fnum), desc="Fundamental matrix estimation for each frame pair"):
            xi = P[i]                         # (2, A)
            vi = np.all(np.isfinite(xi), axis=0)
            for j in range(i + 1, Fnum):
                xj = P[j]                     # (2, A)
                vj = np.all(np.isfinite(xj), axis=0)
                common = vi & vj
                n_common = int(np.count_nonzero(common))
                if n_common < max(8, min_common):
                    continue

                xi_c = xi[:, common]          # (2, N)
                xj_c = xj[:, common]          # (2, N)

                # RANSAC normalized 8-point for F (pixel domain)
                F_ij, _ = TopoUtils.eight_point_normalized_RANSAC(
                    xi_c, xj_c, calibrated=False
                )
                if F_ij is None:
                    continue

                Fmap[(i, j)] = F_ij

                # Store Sampson errors for diagnostics (pixels^2)
                xih = np.vstack([xi_c, np.ones((1, xi_c.shape[1]), dtype=np.float64)])
                xjh = np.vstack([xj_c, np.ones((1, xj_c.shape[1]), dtype=np.float64)])
                Fx1  = F_ij @ xih
                Ftx2 = F_ij.T @ xjh
                numer = np.sum(xjh * Fx1, axis=0)
                denom = Fx1[0]**2 + Fx1[1]**2 + Ftx2[0]**2 + Ftx2[1]**2 + 1e-12
                F_sampson[(i, j)] = (numer**2) / denom  # shape (N,)

        # ---- 2) Fill missing pixels by intersecting epipolar lines
        for a in tqdm(range(A), desc="Filling missing values for each AnyThing"):
            vis_frames = np.where(np.all(np.isfinite(P[:, :, a]), axis=1))[0]
            if vis_frames.size == 0:
                continue

            for j in range(Fnum):
                if j in vis_frames:
                    continue

                lines = []
                for i in vis_frames:
                    if i == j:
                        continue
                    # Get F_ij in the right direction
                    if (i, j) in Fmap:
                        F_ij = Fmap[(i, j)]
                    elif (j, i) in Fmap:
                        F_ij = Fmap[(j, i)].T
                    else:
                        continue

                    xi = P[i, :, a]
                    if not np.all(np.isfinite(xi)):
                        continue

                    xih = np.array([xi[0], xi[1], 1.0], dtype=np.float64)
                    l_j = F_ij @ xih
                    if np.allclose(l_j[:2], 0):
                        continue
                    lines.append(l_j.reshape(3, 1))

                if len(lines) < min_refs:
                    continue

                L = np.hstack(lines)  # (3, N_lines)

                p_j, _ = TopoUtils.solve_point_from_lines(L)

                if np.all(np.isfinite(p_j)):
                    P_out[j, :, a] = p_j.astype(np.float64)

        return P_out, Fmap, F_sampson

    @staticmethod
    def eight_point_fixed_sample_RANSAC(pts1: np.ndarray,
                                        pts2: np.ndarray,
                                        calibrated: bool = False,
                                        *,
                                        n_iter: int = 500,
                                        thresh: float = 5,
                                        min_inliers: int = 12,
                                        rng: Optional[np.random.Generator] = None
                                        ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Estimates matrix (E or F) using RANSAC with a strict 8-point sample strategy.
        NO RE-FITTING step is performed. The model is derived strictly from the sampled 8 points.

        Args:
            pts1 (np.ndarray): (2, N) corresponding points in image 1.
            pts2 (np.ndarray): (2, N) corresponding points in image 2.
            calibrated (bool): If True, estimate an Essential matrix using normalized camera
                coordinates; otherwise, estimate a Fundamental matrix using pixel coordinates.
            n_iter (int): Maximum number of RANSAC iterations.
            thresh (float): Inlier threshold on squared Sampson error, in squared pixel units or
                squared normalized-coordinate units, depending on calibrated.
            min_inliers (int): Minimum number of inliers required to accept a model.
            rng (Optional[np.random.Generator]): Random number generator.

        Returns:
            Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
                Estimated (3,3) matrix and (N,) boolean inlier mask, or (None, None) if no acceptable
                model is found.

        Notes:
            The default threshold of 5 is intended for pixel coordinates.
            When calibrated=True, provide an appropriate threshold for normalized camera coordinates.
        """
        # ---- Checks
        if pts1.ndim != 2 or pts2.ndim != 2 or pts1.shape[0] != 2 or pts2.shape[0] != 2:
            raise ValueError("pts1 and pts2 must be shape (2,N).")
        if pts1.shape[1] != pts2.shape[1]:
            raise ValueError("pts1 and pts2 must have the same number of correspondences.")
        if not np.all(np.isfinite(pts1)) or not np.all(np.isfinite(pts2)):
            raise ValueError("All input points must be finite (no NaNs / Infs).")

        N = pts1.shape[1]
        sample_size = 8

        if N < max(sample_size, min_inliers):
            return None, None

        if rng is None:
            rng = np.random.default_rng()
        assert rng is not None

        best_E: Optional[np.ndarray] = None
        best_inliers: Optional[np.ndarray] = None
        best_count = 0
        best_mean_error = np.inf

        idx_all = np.arange(N)

        # Cap iterations
        cap_comb = math.comb(N, sample_size) if N < 50 else math.inf
        n_iter = int(min(n_iter, cap_comb if cap_comb > 0 else n_iter))

        if calibrated:
            eight_point_estimator = TopoUtils.eight_point_normalized_essential
        else:
            eight_point_estimator = TopoUtils.eight_point_normalized_fundamental

        for _ in range(n_iter):
            # 1) Minimal sample (Strictly 8 points)
            sample = rng.choice(idx_all, size=sample_size, replace=False)

            # Estimate model strictly from these 8 points
            E_sample = eight_point_estimator(pts1[:, sample], pts2[:, sample])

            if E_sample is None or not np.all(np.isfinite(E_sample)):
                continue

            # 2) Score by Sampson error
            errs = TopoUtils.sampson_error(E_sample, pts1, pts2)
            errs = np.where(np.isfinite(errs), errs, np.inf)

            inliers = errs < thresh
            count = int(np.count_nonzero(inliers))

            if count < max(min_inliers, sample_size):
                continue

            current_mean_error = np.mean(errs[inliers])

            # 3) Update Best (NO RE-FITTING)
            if count > best_count or (
                best_E is not None and
                count == best_count and
                current_mean_error < best_mean_error
            ):
                best_count = count
                best_E = E_sample
                best_inliers = inliers
                best_mean_error = current_mean_error

            if best_count == N:
                break

        if best_E is None or best_count < max(min_inliers, sample_size):
            return None, None

        return best_E, best_inliers

    @staticmethod
    def fill_single_frame_via_fixed_sample(
        memory: np.ndarray,
        proposal: np.ndarray,
        *,
        min_common_initial: int = 12,
        min_refs: int = 2
    ) -> np.ndarray:
        """
        Reconstructs a proposal frame using an initial set of eight
        anchor points selected from RANSAC inlier statistics.

        If fewer than eight anchors are identified, returns the original
        proposal unchanged.

        If eight anchors are selected but no valid initial Fundamental
        matrix can be estimated from them, the method returns an array
        containing only those eight anchor observations; all other
        coordinates remain NaN.

        Otherwise, preserves only the selected anchor points and
        estimates the remaining coordinates using epipolar geometry.
        Fundamental matrices are initially estimated from the anchors
        and subsequently refined using an expanding set of reconstructed
        points until no further coordinates can be recovered.

        Unreconstructed coordinates remain NaN.

        Args:
            memory (np.ndarray): (F_mem, 2, A) array of coordinates
                from the reference frames. May contain NaNs.
            proposal (np.ndarray): (2, A) or (1, 2, A) array
                containing the proposal frame coordinates.
            min_common_initial (int): Minimum number of common
                observations required for initial RANSAC estimation.
                At least 8 are required regardless of this value.
            min_refs (int): Minimum number of epipolar lines
                required to attempt reconstructing a point.

        Returns:
            np.ndarray: (2, A) array containing the selected anchors
                and successfully reconstructed coordinates. Unreconstructed
                coordinates remain NaN. If fewer than eight anchors are
                identified, returns a copy of the original proposal.
        """
        from collections import Counter

        # ---- 0. Input Prep
        assert memory.ndim == 3 and memory.shape[1] == 2
        F_mem, _, A = memory.shape

        if proposal.ndim == 3:
            if proposal.shape != (1,2,A):
                raise ValueError("proposal must have shape (2, A) or (1,2,A).")
            proposal_2A = proposal[0]

        elif proposal.ndim == 2:
            if proposal.shape != (2,A):
                raise ValueError("proposal must have shape (2,A) or (1,2,A).")
            proposal_2A = proposal

        else:
            raise ValueError("proposal must have shape (2,A) or (1,2,A).")

        # ---- 1. Feed proposal to eight_point_fixed_sample_RANSAC for each frame
        # to gather inlier statistics.
        inlier_indices_counter = Counter()

        for i in range(F_mem):
            Xi = memory[i]

            # Check overlap
            vi = np.all(np.isfinite(Xi), axis=0)
            vp = np.all(np.isfinite(proposal_2A), axis=0)
            common_mask = vi & vp
            common_indices = np.where(common_mask)[0]

            if len(common_indices) < max(8, min_common_initial):
                continue

            Xi_c = Xi[:, common_indices]
            Xp_c = proposal_2A[:, common_indices]

            _, inlier_mask = TopoUtils.eight_point_fixed_sample_RANSAC(
                Xi_c, Xp_c, calibrated=False, min_inliers=8
            )

            if inlier_mask is not None:
                # Map mask back to global indices and update counter
                valid_global_indices = common_indices[inlier_mask]
                inlier_indices_counter.update(valid_global_indices)

        # ---- 2. Select set of Best 8 points by frequency
        most_common = inlier_indices_counter.most_common(8)

        # If 8 anchors cannot be found, cannot proceed with reconstruction.
        # Fallback: Return original proposal
        if len(most_common) < 8:
            return proposal_2A.copy()

        best_8_indices = [idx for idx, _ in most_common]

        # ---- 3. Initialize Estimation Buffer
        estimation = np.full_like(proposal_2A, np.nan, dtype=np.float64)
        estimation[:, best_8_indices] = proposal_2A[:, best_8_indices]

        # Helper to solve missing points
        def _solve_missing(current_est, f_map):
            next_est = current_est.copy()
            for a in range(A):
                # If already estimated/valid, skip
                if np.all(np.isfinite(next_est[:, a])):
                    continue

                lines = []
                for i, F_i in f_map.items():
                    Xi = memory[i, :, a]
                    if not np.all(np.isfinite(Xi)):
                        continue

                    xih = np.array([Xi[0], Xi[1], 1.0])
                    l_p = F_i @ xih
                    if np.allclose(l_p[:2], 0):
                        continue
                    lines.append(l_p.reshape(3, 1))

                if len(lines) >= min_refs:
                    L = np.hstack(lines)
                    p_hat, _ = TopoUtils.solve_point_from_lines(L)
                    if np.all(np.isfinite(p_hat)):
                        next_est[:, a] = p_hat
            return next_est

        # ---- 4. Initial F Estimation (using ONLY the 8 anchors)
        F_map_initial = {}
        for i in range(F_mem):
            Xi = memory[i]

            pts_mem_8 = Xi[:, best_8_indices]
            pts_est_8 = estimation[:, best_8_indices]

            # Fundamental matrix needs all 8 points to be finite
            if not np.all(np.isfinite(pts_mem_8)):
                continue

            # Direct calculation (No RANSAC)
            F_i = TopoUtils.eight_point_normalized_fundamental(pts_mem_8, pts_est_8)

            if F_i is not None:
                F_map_initial[i] = F_i

        if not F_map_initial:
            return estimation # Returns only the 8 points

        # ---- 5. Initial Solve (reconstruct points based on anchors)
        estimation = _solve_missing(estimation, F_map_initial)

        # ---- 6. Iterative Expansion
        # Loop: Standard RANSAC -> Estimate -> Repeat
        while True:
            current_coverage = np.count_nonzero(np.all(np.isfinite(estimation), axis=0))
            if current_coverage == A:
                break

            F_map_refined = {}
            for i in range(F_mem):
                Xi = memory[i]

                # Check overlap with current estimation
                vi = np.all(np.isfinite(Xi), axis=0)
                vp = np.all(np.isfinite(estimation), axis=0)
                common_mask = vi & vp
                common_indices = np.where(common_mask)[0]

                if len(common_indices) < 8:
                    continue

                Xi_c = Xi[:, common_indices]
                Xp_c = estimation[:, common_indices]

                # Standard RANSAC
                F_i, _ = TopoUtils.eight_point_normalized_RANSAC(
                    Xi_c, Xp_c, calibrated=False, min_samples=8, min_inliers=8
                )

                if F_i is not None:
                    F_map_refined[i] = F_i

            # Solve with new F matrices
            new_estimation = _solve_missing(estimation, F_map_refined)

            new_coverage = np.count_nonzero(np.all(np.isfinite(new_estimation), axis=0))

            if new_coverage <= current_coverage:
                break

            estimation = new_estimation

        return estimation
