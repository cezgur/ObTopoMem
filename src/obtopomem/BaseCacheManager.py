# Copyright 2026 Ata Baran Fırat
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import pickle
import tempfile
from abc import ABC, abstractmethod, ABCMeta
from typing import Generic, TypeVar, Optional, Dict, Any, Type, ClassVar
from collections import OrderedDict

T = TypeVar("T", bound="Cacheable")


class __CacheableMeta(ABCMeta):
    def __new__(mcls, name, bases, ns, **kw):
        cls = super().__new__(mcls, name, bases, ns, **kw)
        return cls

    def __call__(cls, *args, **kwargs):
        obj = super().__call__(*args, **kwargs)
        if hasattr(obj, 'cache_manager') and obj.cache_manager is not None:
            obj.cache_manager.cache(obj)
        return obj


class Cacheable(ABC, metaclass=__CacheableMeta):
    cache_manager: ClassVar[BaseCacheManager[Cacheable]]

    @property
    @abstractmethod
    def id(self) -> str: ...
    @id.setter
    def id(self, new_id: str) -> None: ...

    @property
    @abstractmethod
    def is_dirty(self) -> bool: ...
    @is_dirty.setter
    def is_dirty(self, new_is_dirty: bool) -> None: ...

    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        """Serializes the object to a dictionary."""
        ...

    @classmethod
    @abstractmethod
    def from_dict(cls: Type[T], *args: Any, **kwargs: Any) -> T:
        """Creates an instance of the class from a dictionary."""
        ...


class BaseCacheManager(ABC, Generic[T]):
    """
    An abstract base class for managing the caching of objects to disk.
    It handles the creation of cache directories and the generic logic for
    loading and saving objects.
    """
    def __init__(self,
                 cache_dir: str,
                 cacheableClass: Type[Cacheable],
                 use_cache_disk: bool = True,
                 max_size: Optional[int] = None):
        cacheableClass.cache_manager = self # type: ignore[assignment]
        self.cache_dir = os.path.join(cache_dir, cacheableClass.__name__+"s")
        self._in_memory_cache: OrderedDict[str, T] = OrderedDict()
        self.use_cache_disk = use_cache_disk
        self.max_size = max_size
        if self.use_cache_disk and not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir, exist_ok=True)

    @abstractmethod
    def _load_from_file(self, filepath: str) -> T:
        """
        Provides the logic for deserializing an object from a file.
        """
        ...

    @abstractmethod
    def _get_filepath(self, obj_id: str) -> str:
        """
        Constructs the specific filepath for a given object ID.
        """
        ...

    def get(self, obj_id: str, optional=True) -> Optional[T]:
        """
        Retrieves an object by its ID, from memory or disk.
        """
        if obj_id in self._in_memory_cache:
            self._in_memory_cache.move_to_end(obj_id)
            return self._in_memory_cache[obj_id]

        if self.use_cache_disk:
            filepath = self._get_filepath(obj_id)
            if not os.path.exists(filepath):
                if not optional:
                    raise FileNotFoundError(f"Object was not loaded! ID: {obj_id}")
                return None

            obj = self._load_from_file(filepath)
            self._in_memory_cache[obj_id] = obj
            if self.max_size is not None and len(self._in_memory_cache) > self.max_size:
                self.__evict_LRU_memory()
            return obj

        else:
            if not optional:
                raise FileNotFoundError(f"Object was not loaded! ID: {obj_id}")
            return None

    def __evict_LRU_memory(self) -> Optional[T]:
        if not self._in_memory_cache:
            return None

        oldest_key = next(iter(self._in_memory_cache))
        evicted_obj = self._in_memory_cache[oldest_key]

        # Preserve the in-memory object if persistence fails.
        self.__save(evicted_obj)
        del self._in_memory_cache[oldest_key]
        return evicted_obj

    def __save(self, obj: T) -> None:
        """Atomically saves a dirty object to disk."""
        if not obj.is_dirty or not self.use_cache_disk:
            return

        filepath = self._get_filepath(obj.id)
        fd, temp_path = tempfile.mkstemp(
            dir=os.path.dirname(filepath), prefix=".cache_", suffix=".tmp"
        )
        try:
            with os.fdopen(fd, "wb") as f:
                pickle.dump(obj.to_dict(), f)
                f.flush()
                os.fsync(f.fileno())

            os.replace(temp_path, filepath)
        except Exception:
            if os.path.exists(temp_path):
                os.remove(temp_path)
            raise

        obj.is_dirty = False

    def cache(self, obj: T) -> Optional[T]:
        self._in_memory_cache[obj.id] = obj
        self._in_memory_cache.move_to_end(obj.id)

        evicted_obj: Optional[T] = None
        if self.max_size is not None and len(self._in_memory_cache) > self.max_size:
            evicted_obj = self.__evict_LRU_memory()

        if obj.is_dirty and self.use_cache_disk:
            self.__save(obj)

        return evicted_obj

    def evict_all(self):
        while self._in_memory_cache:
            self.__evict_LRU_memory()
