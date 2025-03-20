import ctypes
import logging


_logger = logging.getLogger(__name__)

try:
    _libc = ctypes.CDLL("libc.so.6")
except:
    _logger.exception("Failed to load libc.so.6")
    raise

try:
    _madvise = _libc.madvise
    _madvise.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
    _madvise.restype = ctypes.c_int
except:
    _logger.exception("Failed to load libc.madvise")
    raise

MADV_DONTNEED = 4


def free_memory(start_ptr: int, end_ptr: int):
    """free memory from start_ptr to end_ptr

    start_ptr and end_ptr must be aligned to the page boundary.
    """

    nbytes = end_ptr - start_ptr
    e = _madvise(start_ptr, nbytes, MADV_DONTNEED)
    return e
