# Copyright 2026 Ata Baran Fırat
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
import functools
from typing import Any, Dict, Optional, Tuple, TypeVar, Iterable
from abc import ABC, abstractmethod

import numpy as np

from .VideoContainer import VideoContainer
from .ImagePoint import KeyPoint, KeyPointCacheManager
from .BaseCacheManager import Cacheable

from .wOmniGlue import wOmniGlue


T_FrameDescriptorBase = TypeVar("T_FrameDescriptorBase", bound="FrameDescriptorBase")

def requires_womniglue(func):
    """
    Decorator to ensure that a method has access to a wOmniGlue instance before proceeding.
    """
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        if self.womniglue is None:
            raise ValueError(f"Method '{func.__name__}' requires a wOmniGlue instance, but it is not available.")
        return func(self, *args, **kwargs)
    return wrapper


class FrameDescriptorBase(Cacheable, ABC):
    logger = logging.getLogger(__name__)

    DEFAULT_MAX_DIM_FOR_PROCESSING = 1028
    DEFAULT_MAX_KEYPOINTS = 1024

    @staticmethod
    def create_id(video_id: str, frame_idx: int) -> str:
        return video_id + "-ind" + str(frame_idx)

    def __init__(self,
                 video_container: VideoContainer,
                 frame_idx: int,
                 womniglue: Optional[wOmniGlue],
                 kp_manager: Optional[KeyPointCacheManager],
                 max_dim_for_processing: Optional[int] = None,
                 max_keypoints: Optional[int] = None,
                 is_dirty: bool = True,
                 id: Optional[str] = None,
                 **kwargs):
        self.videoContainer = video_container
        self.video_id = self.videoContainer.id
        self.__frame_idx = frame_idx
        self.kp_manager = kp_manager
        if id is not None:
            self._id = id
        else:
            self._id = self.create_id(self.video_id, self.frame_idx)

        self.womniglue: Optional[wOmniGlue] = womniglue

        self._max_dim_for_processing = (
            max_dim_for_processing if max_dim_for_processing is not None
            else self.DEFAULT_MAX_DIM_FOR_PROCESSING
        )

        self._max_keypoints = (
            max_keypoints if max_keypoints is not None
            else self.DEFAULT_MAX_KEYPOINTS
        )

        self._is_dirty = is_dirty

        self.dino_patch_size = wOmniGlue.DINO_PATCH_SIZE
        self.dino_feature_dim = wOmniGlue.DINO_FEATURE_DIM

        # Caches
        self._keypoint_ids: Optional[Tuple[str, ...]] = None
        self._DINODescriptors: Optional[np.ndarray] = None
        self._preprocessing_params: Optional[Tuple[Tuple[int, int], float, float]] = None
        self._preprocessed_frame: Optional[np.ndarray] = None

        # Computation flags
        self.dino_descriptors_computed = False

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
    def frame_idx(self) -> int:
        return self.__frame_idx

    @property
    def preprocessing_params(self) -> Tuple[Tuple[int, int], float, float]:
        """
        Returns:
            ((proc_h, proc_w), scale_w, scale_h)
        Where scale_* map preprocessed coordinates to the local (full frame or ROI).
        """
        if self._preprocessing_params is not None:
            return self._preprocessing_params

        raise RuntimeError(f"FrameDescriptor does not contain preprocessing params.")

    @abstractmethod
    def _preprocess_frame(self) -> np.ndarray:
        """
        Returns the preprocessed frame (full or ROI), potentially resized as per preprocessing params.
        """
        ...

    @property
    def preprocessed_frame(self) -> np.ndarray:
        if self._preprocessed_frame is None:
            self._preprocessed_frame = self._preprocess_frame()
        return self._preprocessed_frame

    @requires_womniglue
    def _compute_dino_patch_size(self) -> Tuple[float, float]:
        assert self.womniglue is not None
        frame_shape = self.preprocessed_frame.shape[:2]
        return self.womniglue.get_dino_patch_size_in_image_coords(*frame_shape)

    @property
    def keypoint_ids(self) -> Tuple[str, ...]:
        """
        Build KeyPoint objects in original image coordaintes from processed SP Features.
        """
        if self._keypoint_ids is not None:
            return self._keypoint_ids
        return ()

    @requires_womniglue
    def _compute_dino_features(self) -> np.ndarray:
        """
        Compute dense DINO features for the preprocessed frame (full or ROI).
        """
        assert self.womniglue is not None
        frame = self.preprocessed_frame
        return np.array(self.womniglue.getDINOFeatures(frame), copy=True)

    @property
    def keypoints(self) -> Iterable[KeyPoint]:
        if self.kp_manager is not None:
            for kp_id in self.keypoint_ids:
                kp = self.kp_manager.get(kp_id, optional=False)
                if kp is not None:
                    yield kp

    @abstractmethod
    def _preprocessed_to_original_coords(self, kpts_processed: np.ndarray) -> np.ndarray:
        """
        Map processed-frame keypoint coords (N, 2) -> original image coords (N,2).
        Implement ROI shift here if applicable.
        """
        ...

    @abstractmethod
    def _original_to_preprocessed_indices(self,
                                          kp_locs_orig: np.ndarray,
                                          proc_h: int,
                                          proc_w: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute integer indices into the preprocessed-frame DINO map from original-image
        keypoint coordinates. Must handle ROI shift and scaling appropriately.

        Returns:
            (x_idx, y_idx) both int arrays length N, clipped to [0, proc_w-1] / [0, proc_h-1]
        """
        ...

    def _compute_dino_descriptors_from_aggregated_map(self) -> np.ndarray:
        """
        Lookup DINO descriptors by indexing the preprocessed-frame DINO feature map
        at the keypoint locations converted to preporcessed-frame indices.
        """
        dino_map = self._compute_dino_features()
        (proc_h, proc_w), _, _ = self.preprocessing_params

        kp_locs_orig = np.array([kp.loc for kp in self.keypoints], dtype=float)
        if kp_locs_orig.size == 0:
            if self.dino_feature_dim is None:
                raise ValueError("dino_feature_dim is None, cannot create empty descriptor array.")
            return np.empty((0, int(self.dino_feature_dim)), dtype=np.float32)

        # Get indices in the preprocessed frame coordinate system
        x_idx_preproc_frame, y_idx_preproc_frame = self._original_to_preprocessed_indices(kp_locs_orig, proc_h, proc_w)

        # Scale these indices to the DINO feature map's coordinate system.
        # dino_map.shape[0] is h_feat, dino_map.shape[1] is w_feat
        h_feat, w_feat, _ = dino_map.shape

        # Calculate scaling factors from preprocessed frame dimensions to DINO feature map dimensions.
        scale_h_to_dino_feat = h_feat / proc_h if proc_h > 0 else 0
        scale_w_to_dino_feat = w_feat / proc_w if proc_w > 0 else 0

        x_idx_dino_feat = np.clip(np.rint(x_idx_preproc_frame * scale_w_to_dino_feat).astype(int), 0, w_feat - 1)
        y_idx_dino_feat = np.clip(np.rint(y_idx_preproc_frame * scale_h_to_dino_feat).astype(int), 0, h_feat - 1)

        return dino_map[y_idx_dino_feat, x_idx_dino_feat]

    @property
    def DINODescriptorsRowVec(self) -> np.ndarray:
        """
        Lazily computes DINO descriptors for the keypoints by direct lookup
        on the DINO map. Also attaches each descriptor to its KeyPoint.
        """
        if not self.dino_descriptors_computed:
            d = self._compute_dino_descriptors_from_aggregated_map()
            kps = self.keypoints
            for i, kp in enumerate(kps):
                kp.dino_descriptor = d[i]
            self._DINODescriptors = d
            self.dino_descriptors_computed = True
            self.is_dirty = True
        assert self._DINODescriptors is not None
        return self._DINODescriptors

    @property
    def DINODescriptorsColVec(self) -> np.ndarray:
        """
        Returns DINODescriptors as a 2D numpy array where each column is a descriptor vector.
        Shape will be (DINO Feature dimenison, Number of KeyPoints).
        """
        return self.DINODescriptorsRowVec.T

    def to_dict(self) -> Dict[str, Any]:
        """
        Common serialization payload, subclasses can 'update()' with their own fields.
        """
        return {
            "id": self.id,
            "video_id": self.video_id,
            "frame_idx": self.frame_idx,
            "keypoint_ids": self.keypoint_ids,
            "dino_descriptors": self._DINODescriptors,
            "preprocessing_params": self.preprocessing_params,
            "cache_max_dim_for_processing": self._max_dim_for_processing,
            "dino_patch_size": self.dino_patch_size,
            "dino_feature_dim": self.dino_feature_dim,
            "dino_descriptors_computed": self.dino_descriptors_computed,
            "cache_max_keypoints": self._max_keypoints
        }

    @staticmethod
    def _rehydrate_caches(instance: "FrameDescriptorBase",
                          data: Dict[str, Any],
                          kp_manager: "KeyPointCacheManager") -> None:
        """
        Populate an instance's caches fromn a serialized dict.
        Subclass should call this, then set any of their own fields.
        """
        # KeyPoints
        instance._keypoint_ids = data.get("keypoint_ids")

        if (dd := data.get("dino_descriptors")) is not None:
            instance._DINODescriptors = dd

        if (pp := data.get("preprocessing_params")) is not None:
            instance._preprocessing_params = pp

        instance.dino_patch_size = data.get("dino_patch_size")
        instance.dino_feature_dim = data.get("dino_feature_dim")

        instance.kp_manager = kp_manager

        instance.is_dirty = False

        # Flags
        instance.dino_descriptors_computed = data.get("dino_descriptors_computed")
