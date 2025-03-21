from dataclasses import dataclass
import logging
from typing import Any, Mapping, Iterator, NamedTuple

import torch

from .mmap import free_memory


_logger = logging.getLogger(__name__)


def _align(p, n=4096):
    """align n bytes boundary (floor)"""
    return p & ~(n - 1)


def _align1(p, n=4096):
    """align n bytes boundary (ceil)"""
    return (p + n - 1) & ~(n - 1)


@dataclass(frozen=True)
class _TensorInfo:
    ptr: int
    """pointer to the tensor data (storage)"""

    nbytes: int
    """data size in bytes"""

    parent: Any | None
    """parent object which contains this tensor"""

    key: Any | None
    """key of the parent object points to this tensor"""

    tensor: torch.Tensor
    """tensor data"""


def _retrieve_all_tensors(item: Any, parent: Any, key: Any | None) -> Iterator[_TensorInfo]:
    """retrieve all tensors from the object"""

    if isinstance(item, dict):
        for k, v in item.items():
            yield from _retrieve_all_tensors(v, item, k)
    elif isinstance(item, (tuple, list)):
        for i, v in enumerate(item):
            yield from _retrieve_all_tensors(item, item, i)
    elif isinstance(item, torch.Tensor):
        ptr = item.data_ptr()
        nbytes = item.numel() * item.dtype.itemsize
        yield _TensorInfo(ptr, nbytes, parent, key, item)


class StreamValue(NamedTuple):
    parent: Any
    key: Any
    tensor: torch.Tensor


def stream(
    obj: Mapping[str, Any],
    memory_limit_mb: int = 1024,
    cpu_page_size: int = 4096,
) -> Iterator[StreamValue]:
    """
    Yields tuples of (parent, key, tensor) from the given state_dict, ensuring that the physical memory usage does not exceed the specified limit.

    Args:
        obj (Mapping[str, Any]): The checkpoint dictionary containing tensors.
        memory_limit_mb (int, optional): The maximum amount of tensor data allowed to remain in physical memory, measured in MiB. Defaults to 1024.
        cpu_page_size (int, optional): The system's memory page size in bytes. Defaults to 4096.

    Yields:
        Iterator[StreamValue]: Tuples of (parent, key, tensor), where `tensor` is always a torch.Tensor, `parent` is the container of `tensor`, and `key` is the key in `parent` such that `parent[key] == tensor`.
    """

    # sort tensors by their pointer
    all_tensors = sorted(_retrieve_all_tensors(obj, None, None), key=lambda x: x.ptr)

    # align to the first tensor's pointer
    start_ptr = _align1(all_tensors[0].ptr, cpu_page_size)

    for x in all_tensors:
        p = x.ptr
        n = x.nbytes

        try:
            yield StreamValue(x.parent, x.key, x.tensor)

        finally:
            # here, [start_ptr .. (p+n)] is on the physical memory

            end_ptr = _align(p + n)
            nbytes = end_ptr - start_ptr  # number of bytes of data on physical memory

            if memory_limit_mb * 1024 * 1024 <= nbytes:
                # free physical memory if it exceeds memory_limit_mb
                e = free_memory(start_ptr, end_ptr)
                if e == 0:
                    start_ptr = end_ptr
                else:
                    _logger.error(
                        f"Failed to call madvise: {e} (key={x.key}, ptr0={start_ptr}, ptr1={end_ptr}, nbytes={nbytes})"
                    )
                    # ignore this region
                    start_ptr = end_ptr
