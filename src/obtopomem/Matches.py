# Copyright 2026 Ata Baran Fırat
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from .ImagePoint import KeyPointCacheManager
from typing import (
    Tuple,
    Dict,
    overload,
    List,
    Optional,
    Iterator,
    Any,
    FrozenSet
)
import uuid
import os
import pickle
from bidict import bidict

from .BaseCacheManager import BaseCacheManager, Cacheable


class Match(Cacheable):
    """
    Represents a correspondence between two KeyPoint objects.
    """
    def __init__(self,
                 keypoint_ids: Tuple[str, str],
                 kp_manager: KeyPointCacheManager,
                 confidence: float,
                 id: Optional[str] = None,
                 is_dirty: bool = True) -> None:
        """
        Initializes a Match object.

        Args:
            keypoint_ids: A tuple containing exactly two KeyPoint ids.
            confidence: A float representing the confidence of this Match.
            id (Optional[str]): An existing ID to assign. If None, a new one is generated.

        Raises:
            ValueError: If the keypoint_ids does not contain exactly distinct two elements.
        """
        if len(keypoint_ids) != 2 or keypoint_ids[0] == keypoint_ids[1]:
            raise ValueError("A Match must consist of exactly two distinct keypoints.")
        self.kp_manager = kp_manager
        self.__keypoint_ids: FrozenSet[str] = frozenset({keypoint_ids[0], keypoint_ids[1]})
        self.__confidence: float = confidence
        self._is_dirty = is_dirty

        kp1 = self.kp_manager.get(keypoint_ids[0], optional=False)
        kp2 = self.kp_manager.get(keypoint_ids[1], optional=False)
        assert kp1 is not None and kp2 is not None, f"At least one of the keypoint is not found in cache."
        self.frame_ids: Dict[str, str] = {kp1.id: kp1.frame_id,
                                          kp2.id: kp2.frame_id}

        self._id = id if id is not None else str(uuid.uuid4())

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
    def keypoint_ids(self) -> FrozenSet[str]:
        return self.__keypoint_ids

    @property
    def confidence(self) -> float:
        """The confidence score associated with this Match."""
        return self.__confidence
    @confidence.setter
    def confidence(self, new_confidence) -> None:
        self.__confidence = new_confidence
        self.is_dirty = True

    def __len__(self) -> int:
        """Returns the number of keypoint ids stored in the match (always 2)."""
        return len(self.__keypoint_ids)

    def __repr__(self) -> str:
        """Returns a developer-friendly string representation of the Match."""
        kp_id1, kp_id2 = self.__keypoint_ids
        return f"Match(keypoints=({kp_id1[:8]}..., {kp_id2[:8]}...), confidence={self.confidence:.4f})"

    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes the Match object to a dictionary by storing KeyPoint IDs.
        """
        return {
            "id": self.id,
            "keypoint_ids": tuple(self.__keypoint_ids),
            "confidence": self.confidence
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any], kp_manager: KeyPointCacheManager) -> Match:
        return cls(
            keypoint_ids=data["keypoint_ids"],
            kp_manager=kp_manager,
            confidence=data["confidence"],
            id=data["id"],
            is_dirty=False
        )


class Matches(Cacheable):
    """
    A collection of Match instances between two specific frames.

    This class acts as a container for multiple Match objects, ensuring that all
    matches within the collection share the same pair of frame IDs. It behaves
    like a list, allowing for iteration, and length checking.
    """
    def __init__(self,
                 match_ids: List[str],
                 frame_ids: Tuple[str, str],
                 match_manager: MatchCacheManager,
                 id: Optional[str] = None,
                 is_dirty: bool = True):
        """
        Initializes a Matches object.

        Args:
            match_ids: A list of Match object IDs.
            frame_ids: A tuple of the two frame IDs this collection connects.
            match_manager: The cache manager for Match objects.
            id (Optional[str]): An existing ID to assign. If None, a new one is generated.
            is_dirty (bool): Initial dirty state.
        Raises:
            ValueError: If the provided matches do not all share the same pair of frame IDs.
        """
        if len(frame_ids) != 2:
            raise ValueError("Matches must be initialized with exactly two frame_ids.")

        self.match_manager = match_manager
        self._match_ids: List[str] = list(match_ids)
        self.__frame_ids: FrozenSet[str] = frozenset(frame_ids)
        self._id: str = id if id is not None else "_".join(sorted(self.__frame_ids))
        self._is_dirty: bool = is_dirty

        self.__keypoint_id_to_match_id_map = {}

        # Create keypoint map for convenience, and verify consistency of contained matches.
        matches: List[Match] = []
        for match_id in match_ids:
            match = self.match_manager.get(match_id, optional=False)
            if match is not None:
                matches.append(match)
                kp1_id, kp2_id = match.keypoint_ids
                self.__keypoint_id_to_match_id_map[kp1_id], self.__keypoint_id_to_match_id_map[kp2_id] = match_id, match_id

        if matches:
            # Verify all matches conform to the same frame ID pair.
            for match in matches:
                current_frame_ids = frozenset(match.frame_ids.values())
                if current_frame_ids != self.__frame_ids:
                    raise ValueError(
                        "All matches in the collection must share the same pair of frame IDs."
                        f"Expected {self.__frame_ids}, but found {current_frame_ids}."
                    )
            keypoint_ids_list = [list(match.keypoint_ids) for match in matches]
            self.__keypoint_id_map = bidict({keypoint_ids[0]: keypoint_ids[1] for keypoint_ids in keypoint_ids_list})
        else:
            self.__keypoint_id_map = bidict()

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
    def frame_ids(self) -> Optional[FrozenSet[str]]:
        """The pair of frame IDs common to all matches in this collection."""
        return self.__frame_ids

    @property
    def match_ids(self) -> List[str]:
        return list(self._match_ids)

    def __len__(self) -> int:
        """Returns the number of matches in the collection."""
        return len(self.match_ids)

    def kp_id_map(self, kp_id: str) -> Optional[str]:
        return self.__keypoint_id_map.get(kp_id,
                                          self.__keypoint_id_map.inv.get(kp_id,
                                                                         None))

    def kp_id_to_match_map(self, kp_id: str) -> Optional[Match]:
        match_id = self.__keypoint_id_to_match_id_map.get(kp_id, None)
        if match_id is None:
            return None
        else:
            return self.match_manager.get(match_id, optional=False)

    @overload
    def __getitem__(self, index: int) -> Match: ...
    @overload
    def __getitem__(self, index: slice) -> List[Match]: ...
    def __getitem__(self, index: int | slice) -> Match | List[Match]:
        """Allows accessing the matches by index or slice."""
        if isinstance(index, slice):
            match_ids = self.match_ids[index]
            matches: List[Match] = []
            for match_id in match_ids:
                match = self.match_manager.get(match_id, optional=True)
                if match is not None:
                    matches.append(match)
            return matches
        else:
            match_id = self.match_ids[index]
            match = self.match_manager.get(match_id, optional=False)
            if match is None:
                raise IndexError(f"Match with id {match_id} not found in cache.")
            return match

    def __iter__(self) -> Iterator[Match]:
        """Allows iteration over the matches."""
        for match_id in self.match_ids:
            match = self.match_manager.get(match_id, optional=False)
            if match is not None:
                yield match

    def __repr__(self) -> str:
        """Returns a developer-friendly string representation of the Matches collection."""
        return f'Matches(count={len(self)}, frame_ids={self.frame_ids})'

    def sum_with(self,
                 other_matches: Matches) -> Optional[str]:
        """
        Combines two Matches objects that share a common intermediate frame,
        creating a new Matches object that links the two uncommon frames.

        Example: self links (A, B), other_matches links (B, C).
        Result links (A, C).

        Args:
            other_matches: The second Matches object to combine with.

        Returns:
            The id of a new Matches object linking the uncommon frames, or None if no
            valid common frame is found or no bridging matches exist.
        """
        if self.frame_ids is None or other_matches.frame_ids is None:
            return None # Cannot combine if frame_ids are not set

        common_frame_ids = self.frame_ids.intersection(other_matches.frame_ids)
        if len(common_frame_ids) != 1:
            return None # No single common frame
        common_frame_id = list(common_frame_ids)[0]

        matches_first_leg = self
        matches_second_leg = other_matches

        # Create mappings for efficient lookup of keypoints in the common frame
        # kp_in_common_frame_id -> (kp_in_uncommon_frame_id, confidence)
        map1: Dict[str, Tuple[str, float]] = {}
        for match in matches_first_leg:
            kp_a_id, kp_b_id = match.keypoint_ids
            if match.frame_ids[kp_a_id] == common_frame_id:
                map1[kp_a_id] = (kp_b_id, match.confidence)
            elif match.frame_ids[kp_b_id] == common_frame_id:
                map1[kp_b_id] = (kp_a_id, match.confidence)
            # else: This should not happen if matches_first_leg is valid for common_frame_id

        map2: Dict[str, Tuple[str, float]] = {}
        for match in matches_second_leg:
            kp_a_id, kp_b_id = match.keypoint_ids
            if match.frame_ids[kp_a_id] == common_frame_id:
                map2[kp_a_id] = (kp_b_id, match.confidence)
            elif match.frame_ids[kp_b_id] == common_frame_id:
                map2[kp_b_id] = (kp_a_id, match.confidence)
            # else: This should not happen if matches_second_leg is valid for common_frame_id

        new_matches_list: List[Match] = []
        # Find bridging keypoints in the common frame
        for kp_common_id, (kp_uncommon1_id, conf1) in map1.items():
            if kp_common_id in map2:
                kp_uncommon2_id, conf2 = map2[kp_common_id]

                # Create a new Match object
                # The new match links kp_uncommon1 and kp_uncommon2
                # Confidence combination: min(conf1, conf2)
                new_confidence = min(conf1, conf2)
                new_match = Match(keypoint_ids=(kp_uncommon1_id, kp_uncommon2_id),
                                  kp_manager=self.match_manager.keypoint_manager,
                                  confidence=new_confidence)
                new_matches_list.append(new_match)

        if not new_matches_list:
            return None # No bridging matches found

        # Create a new Matches object with a temporary, descriptive ID
        match_ids = [match.id for match in new_matches_list]
        first_new_match = new_matches_list[0]
        frame_id_A, frame_id_C = sorted(tuple(first_new_match.frame_ids.values()))
        summed_matches_id = "->".join(["_".join([frame_id_A, frame_id_C]), common_frame_id])
        summed_matches = Matches(match_ids,
                                 frame_ids=(frame_id_A, frame_id_C),
                                 match_manager = self.match_manager,
                                 id=summed_matches_id)

        # Call the method to update the existing direct match based on this new evidence
        self.update_with_summed_evidence(summed_matches)

        return summed_matches_id

    @classmethod
    def update_with_summed_evidence(cls, summed_matches: Matches):
        """
        Updates the confidences of an existing direct match object based on new
        evidence from a summed match. Does not create new matches.
        """
        assert isinstance(cls.cache_manager, MatchesCacheManager)
        if not summed_matches.frame_ids:
            return

        # --- Step 1: Get Direct Matches ---
        frame_id_A, frame_id_C = tuple(summed_matches.frame_ids)
        direct_matches = cls.cache_manager.get_by_frame_pair(frame_id_A, frame_id_C)
        if not direct_matches:
            return

        # --- Build lookup maps from the new evidence for efficient access ---
        summed_evidence_map: Dict[FrozenSet[str], Match] = {m.keypoint_ids: m for m in summed_matches}
        summed_kp_to_keys: Dict[str, List[FrozenSet[str]]] = {k:[] for m in summed_evidence_map for k in m}
        for key in summed_evidence_map:
            for kp_id in key:
                summed_kp_to_keys[kp_id].append(key)

        # --- Step 2: Update confidences of each direct match ---
        # Modify the list of matches from the direct_matches object in-place.
        for match_obj in direct_matches:
            c_final = match_obj.confidence
            key = match_obj.keypoint_ids
            kp1, kp2 = tuple(key)

            # Check for agreeing evidence
            if key in summed_evidence_map:
                c_agreement = summed_evidence_map[key].confidence
                c_final = 1 - (1 - c_final) * (1 - c_agreement)

            # Check for conflicting evidence
            conflicting_keys = (set(summed_kp_to_keys.get(kp1, [])) | set(summed_kp_to_keys.get(kp2, [])))
            if key in conflicting_keys:
                conflicting_keys.remove(key)

            if conflicting_keys:
                c_conflict = 0.0
                for conflict_key in conflicting_keys:
                    c_competitor = summed_evidence_map[conflict_key].confidence
                    c_conflict = 1 - (1 - c_conflict) * (1 - c_competitor)

                if c_conflict >= 1.0:
                    c_final = -1.0 # Disproven
                else:
                    c_final = (c_final - c_conflict) / (1 - c_conflict)

            # Update the confidence on the object
            match_obj.confidence = c_final

        # --- Step 3: Filter and Save ---
        # Create a new list from the modified direct_matches, keeping only survivors.
        final_match_ids_list = [m.id for m in direct_matches if m.confidence > 0]

        # Create the new Matches object, which will overwrite the old one in the cache.
        frame_ids = direct_matches.frame_ids
        assert frame_ids is not None
        frame_ids = tuple(frame_ids)
        assert len(frame_ids) == 2

        _ = Matches(final_match_ids_list,
                    frame_ids=frame_ids,
                    match_manager=summed_matches.match_manager,
                    id=direct_matches.id, is_dirty=True)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "frame_ids": self.frame_ids,
            "match_ids": self.match_ids,
        }

    @classmethod
    def from_dict(cls,
                  data: Dict[str, Any],
                  match_manager: MatchCacheManager):
        frame_ids = data.get("frame_ids")
        if frame_ids is None:
            raise ValueError("Cannot deserialize Matches object: 'frame_ids' is missing from cached data.")
        return cls(match_ids = data["match_ids"],
                   frame_ids=tuple(frame_ids),
                   match_manager=match_manager,
                   id=data.get("id"),
                   is_dirty=False)


class MatchCacheManager(BaseCacheManager[Match]):
    def __init__(self, cache_dir: str, keypoint_manager: KeyPointCacheManager, use_cache_disk: bool = True):
        super().__init__(cache_dir, Match, use_cache_disk=use_cache_disk)
        self.keypoint_manager = keypoint_manager

    def _get_filepath(self, obj_id: str) -> str:
        return os.path.join(self.cache_dir, f"{obj_id}.pkl")

    def _load_from_file(self, filepath: str) -> Match:
        with open(filepath, "rb") as f:
            data = pickle.load(f)

        return Match.from_dict(data, self.keypoint_manager)


class MatchesCacheManager(BaseCacheManager[Matches]):
    def __init__(self, cache_dir: str, match_manager: MatchCacheManager, use_cache_disk: bool = True, max_size: Optional[int] = 25): # Added max_size with default 25
        super().__init__(cache_dir, Matches, use_cache_disk=use_cache_disk, max_size=max_size) # Passed max_size
        self.match_manager = match_manager

    def get_by_frame_pair(self, frame1_id: str, frame2_id: str, optional=True) -> Optional[Matches]:
        potential_matches_id = "_".join(sorted([frame1_id, frame2_id]))
        return self.get(potential_matches_id, optional=optional)

    def _get_filepath(self, obj_id: str) -> str:
        return os.path.join(self.cache_dir, f"{obj_id}.pkl")

    def _load_from_file(self, filepath: str) -> Matches:
        with open(filepath, "rb") as f:
            data = pickle.load(f)
        return Matches.from_dict(data, self.match_manager)
