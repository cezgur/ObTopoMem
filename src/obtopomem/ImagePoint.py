# Copyright 2026 Ata Baran Fırat
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, List, Iterable
import uuid
import os
import pickle
import numpy as np

from .BaseCacheManager import BaseCacheManager, Cacheable
from .wOmniGlue import wOmniGlue


class ImagePoint(Cacheable):
    """
    Represents an image point from a specific frame of a video or an image.

    Attributes:
        id (str): A unique identifier for the image point.
        loc (Tuple[float, float]): The (x, y) coordinates of the point.
        frame_id (str): The ID of the FrameDescirptor this point belongs to.
    """
    def __init__(self,
                 loc: Tuple[float, float],
                 frame_id: str,
                 id: str | None = None,
                 is_dirty: bool = True):
        """
        Initializes a point object.

        Args:
            loc: A tuple representing the (x, y) coordinates of the point.
            frame_id: The ID of the FrameDescriptor this point belongs to.
            id (str | None): An existing ID to assign. If None, a new one is generated.
        """
        self._id = id if id is not None else str(uuid.uuid4())
        self.loc = (float(loc[0]), float(loc[1]))
        self.frame_id = frame_id
        self._is_dirty = is_dirty

    @property
    def id(self) -> str:
        return self._id
    @id.setter
    def id(self, new_id: str) -> None:
        self._id = new_id

    @property
    def is_dirty(self)-> bool:
        return self._is_dirty
    @is_dirty.setter
    def is_dirty(self, new_is_dirty: bool) -> None:
        self._is_dirty = new_is_dirty

    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes the ImagePoint object to a dictionary.

        Returns:
            Dict[str, Any]: A dictionary representation of the point.
        """
        return {
            "id": self.id,
            "loc": self.loc,
            "frame_id": self.frame_id
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ImagePoint:
        """
        Deserializes a point from a dicrionary.
        """
        return cls(
            loc = tuple(data["loc"]),
            frame_id=data["frame_id"],
            id=data["id"],
            is_dirty=False
        )

    def __eq__(self, other: object) -> bool:
        return isinstance(other, ImagePoint) and self.id == other.id

    def __hash__(self) -> int:
        return hash(self.id)


class KeyPoint(ImagePoint):
    """
    Represents a keypoint detected in a specific frame of a video.

    Attributes:
        id(str): A unique identifier for the keypoint.
        loc (Tuple[float, float]): The (x, y) coordinates of the keypoint.
        sp_descriptor (np.ndarray): The 256-dim SuperPoint descriptor.
        score (float): The confidence score from the SuperPoint detector.
        frame_id (str): The ID of the FrameDescriptor this keypoint belongs to.
        dino_descriptor (Optional[np.ndarray]): The DINOv2 descriptor.
    """
    def __init__(self,
                 loc: Tuple[float, float],
                 sp_descriptor: np.ndarray,
                 score: float,
                 frame_id: str,
                 id: str | None = None,
                 is_dirty: bool = True):
        """
        Initializes a KeyPoint object.

        Args:
            loc: A tuple representing the (x, y) coordinates of the keypoint.
            sp_descriptor: The SuperPoint feature descirptor.
            score: The confidence score of the keypoint.
            frame_id: The ID of the FrameDescriptor this keypoint belongs to.
        """
        super().__init__(loc, frame_id, id, is_dirty)

        self.sp_descriptor = np.ascontiguousarray(np.asarray(sp_descriptor).reshape(-1), dtype=np.float32)
        self.score = float(score)
        self._dino_descriptor: Optional[np.ndarray] = None

    @property
    def dino_descriptor(self) -> Optional[np.ndarray]:
        return self._dino_descriptor
    @dino_descriptor.setter
    def dino_descriptor(self, value: Optional[np.ndarray]):
        if value is None:
            self._dino_descriptor = None
        else:
            # ensure 1D, contiguous, float32
            arr = np.ascontiguousarray(np.asarray(value).reshape(-1), dtype=np.float32)
            self._dino_descriptor = arr
        self.is_dirty = True

    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes the KeyPoint object to a dictionary.

        Returns:
            Dict[str, Any]: A dictionary representation of the KeyPoint.
        """
        to_dicted: Dict[str, Any] = super().to_dict()
        to_dicted.update({"sp_descriptor": self.sp_descriptor,
                          "score": self.score,
                          "dino_descriptor": self.dino_descriptor})
        return to_dicted

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> KeyPoint:
        """Deserializes a keypoint from a dictionary."""
        inst = cls(
            loc=tuple(data["loc"]),
            sp_descriptor=np.asarray(data["sp_descriptor"]),
            score=float(data["score"]),
            frame_id=data["frame_id"],
            id=data["id"],
            is_dirty=False,
        )
        # setter normalizes dtype/shape and sets dirty; then reset
        inst.dino_descriptor = data.get("dino_descriptor")
        inst.is_dirty = False
        return inst

    def __eq__(self, other: object) -> bool:
        return isinstance(other, KeyPoint) and self.id == other.id

    def __hash__(self) -> int:
        return hash(self.id)


class KeyPoints(Cacheable):
    def __init__(self,
                 kp_ids: List[str],
                 kp_manager: KeyPointCacheManager,
                 id: Optional[str] = None,
                 is_dirty: bool = True):
        self.kp_manager = kp_manager
        self._kp_ids: List[str] = []
        self._frame_id: Optional[str] = None # Assigned in kp_ids.setter
        self._locs: np.ndarray = np.empty((2,1), dtype=int) # Assigned in kp_ids.setter

        self.kp_ids = kp_ids

        self._id: str = id if id is not None else str(uuid.uuid4())
        self._is_dirty = is_dirty

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
    def is_dirty(self, new_is_dirty) -> None:
        self._is_dirty = new_is_dirty

    @property
    def kp_ids(self) -> List[str]:
        return self._kp_ids
    @kp_ids.setter
    def kp_ids(self, new_kp_ids: List[str]) -> None:
        accepted_ids: List[str] = []
        locs = []
        frame_id: Optional[str] = None

        for kp_id in new_kp_ids:
            kp = self.kp_manager.get(kp_id, optional=False)
            if kp is None or kp.frame_id is None:
                raise ValueError(
                    f"KeyPoint: {kp_id!r} has no frame identity."
                )

            if frame_id is None:
                frame_id = kp.frame_id

            if kp.frame_id == frame_id:
                accepted_ids.append(kp_id)
                locs.append(kp.loc)

        if not accepted_ids:
            raise ValueError("KeyPoints requires at least one KeyPoint.")

        new_locs = np.asarray(locs, dtype=float).T

        self._kp_ids = accepted_ids
        self._frame_id = frame_id
        self._locs = new_locs
        self.is_dirty = True

    @property
    def keypoints(self) -> Iterable[KeyPoint]:
        for kp_id in self.kp_ids:
            if kp := self.kp_manager.get(kp_id, optional=False):
                yield kp

    @property
    def frame_id(self) -> str:
        assert self._frame_id is not None
        return self._frame_id

    @property
    def DINODescriptorsRowVec(self) -> np.ndarray:
        """
        Collects DINO descriptors from the KeyPoint objects stored in this KeyPoints object.
        """
        all_descriptors = []
        for kp in self.keypoints:
            if kp.dino_descriptor is not None:
                all_descriptors.append(kp.dino_descriptor.reshape(1, -1)) # Ensure row vector

        if not all_descriptors:
            dino_feature_dim = wOmniGlue.DINO_FEATURE_DIM
            return np.empty((0, dino_feature_dim), dtype=np.float32)

        # Concatenate all collected descriptor arrays veritcally.
        return np.concatenate(all_descriptors, axis=0)

    @property
    def DINODescriptorsColVec(self) -> np.ndarray:
        return self.DINODescriptorsRowVec.T

    @property
    def locs(self) -> np.ndarray:
        return self._locs

    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes the KeyPoints object to a dictionary.

        Returns:
            Dict[str, Any]: A dictionary representation of the KeyPoints.
        """
        return {"id": self.id,
                "locs": self.locs,
                "kp_ids": self.kp_ids,
                "frame_id": self.frame_id}

    @classmethod
    def from_dict(cls, data: Dict[str, Any], kp_manager: KeyPointCacheManager) -> KeyPoints:
        return cls(kp_ids=data["kp_ids"],
                   kp_manager=kp_manager,
                   id=data["id"],
                   is_dirty=False)


class ImagePointCacheManager(BaseCacheManager[ImagePoint]):
    def __init__(self, cache_dir: str, use_cache_disk: bool = True, max_size: int = 10000):
        super().__init__(cache_dir, ImagePoint, use_cache_disk=use_cache_disk, max_size=max_size)

    def _get_filepath(self, obj_id: str) -> str:
        return os.path.join(self.cache_dir, f"{obj_id}.pkl")

    def _load_from_file(self, filepath: str) -> ImagePoint:
        with open(filepath, "rb") as f:
            data = pickle.load(f)
        return ImagePoint.from_dict(data)


class KeyPointCacheManager(BaseCacheManager[KeyPoint]):
    def __init__(self, cache_dir: str, use_cache_disk: bool = True, max_size: int = 1000):
        super().__init__(cache_dir, KeyPoint, use_cache_disk=use_cache_disk, max_size=max_size)

    def _get_filepath(self, obj_id: str) -> str:
        return os.path.join(self.cache_dir, f"{obj_id}.pkl")

    def _load_from_file(self, filepath: str) -> KeyPoint:
        with open(filepath, "rb") as f:
            data = pickle.load(f)
        return KeyPoint.from_dict(data)


class KeyPointsCacheManager(BaseCacheManager[KeyPoints]):
    def __init__(self,
                 cache_dir: str,
                 kp_manager: KeyPointCacheManager,
                 use_cache_disk: bool = True,
                 max_size: int = 100):
        super().__init__(cache_dir, KeyPoints, use_cache_disk=use_cache_disk, max_size=max_size)
        self.kp_manager: KeyPointCacheManager = kp_manager

    def _get_filepath(self, obj_id: str) -> str:
        return os.path.join(self.cache_dir, f"{obj_id}.pkl")

    def _load_from_file(self, filepath: str) -> KeyPoints:
        with open(filepath, "rb") as f:
            data = pickle.load(f)
        return KeyPoints.from_dict(data, self.kp_manager)
