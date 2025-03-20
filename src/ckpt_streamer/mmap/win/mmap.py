import ctypes
import logging


_logger = logging.getLogger(__name__)


try:
    from ctypes import windll as _windll, wintypes as _wintypes
except:
    _logger.exception("Failed to load windll")
    raise


def free_memory(start_ptr: int, end_ptr: int):
    raise NotImplementedError()
