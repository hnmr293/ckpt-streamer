from dataclasses import dataclass
import logging
from typing import Any, Mapping, Iterator, NamedTuple, Callable

import torch
from torch.nn.modules.module import _IncompatibleKeys

from .mmap import free_memory
from .utils import get_module_and_param


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


def apply_state_dict(
    model: torch.nn.Module,
    state_dict: dict[str, Any],
    converter: Callable[
        [torch.nn.Module, torch.nn.Module, torch.Tensor], torch.Tensor
    ] = lambda root_module, current_module, x: x,
    strict: bool = True,
    assign: bool = False,
    memory_limit_mb: int = 1024,
    cpu_page_size: int = 4096,
) -> _IncompatibleKeys:
    """
    Applies a state_dict to a model in a streaming fashion, ensuring that the memory usage does not exceed the specified limit.

    Args:
        model (torch.nn.Module): An instance of torch.nn.Module.
        state_dict (dict[str, Any]): A checkpoint dictionary containing tensors. Must be created by `torch.load(..., map_location='cpu', mmap=True), or causes SEGV`.
        converter (Callable[[torch.nn.Module, torch.nn.Module, torch.Tensor], torch.Tensor], optional): A function to convert tensors. Defaults to a no-op lambda function.
        strict (bool, optional): Whether to strictly enforce that the keys in state_dict match the keys returned by model.state_dict(). Defaults to True.
        assign (bool, optional): Whether to assign the tensor as a torch.nn.Parameter. Defaults to False.
        memory_limit_mb (int, optional): Memory limit in MiB. Defaults to 1024.
        cpu_page_size (int, optional): System's memory page size in bytes. Defaults to 4096.

    Returns:
        _IncompatibleKeys: An object containing missing_keys and unexpected_keys.
    """
    missing_keys: set[str] = set(model.state_dict().keys())
    unexpected_keys: list[str] = []
    error_msgs: list[str] = []

    for _, input_key, input_tensor in stream(
        state_dict,
        memory_limit_mb=memory_limit_mb,
        cpu_page_size=cpu_page_size,
    ):
        with torch.no_grad():
            try:
                target_module, target_param_key, target_param = get_module_and_param(model, input_key)
            except Exception as e:
                unexpected_keys.append(input_key)
                continue

            converted_tensor = converter(model, target_module, input_tensor)
            if converted_tensor.data_ptr() == input_tensor.data_ptr():
                # model must not have mmaped tensors
                converted_tensor = input_tensor.clone()
            input_tensor = converted_tensor

            if assign:
                if not isinstance(input_tensor, torch.nn.Parameter):
                    input_tensor = torch.nn.Parameter(
                        input_tensor,
                        requires_grad=target_param.requires_grad,
                    )
                else:
                    input_tensor.requires_grad_(target_param.requires_grad)
                setattr(target_module, target_param_key, input_tensor)
            else:
                target_param.copy_(input_tensor)

            missing_keys.remove(input_key)

    if strict:
        if len(unexpected_keys) > 0:
            error_msgs.insert(
                0,
                "Unexpected key(s) in state_dict: {}. ".format(
                    ", ".join(f'"{k}"' for k in unexpected_keys),
                ),
            )
        if len(missing_keys) > 0:
            error_msgs.insert(
                0,
                "Missing key(s) in state_dict: {}. ".format(
                    ", ".join(f'"{k}"' for k in missing_keys),
                ),
            )

    if len(error_msgs) > 0:
        raise RuntimeError(
            "Error(s) in loading state_dict for {}:\n\t{}".format(
                model.__class__.__name__,
                "\n\t".join(error_msgs),
            )
        )

    return _IncompatibleKeys(list(missing_keys), unexpected_keys)
