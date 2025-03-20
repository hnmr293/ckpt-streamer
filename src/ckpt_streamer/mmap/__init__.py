def __check_posix():
    import ctypes

    try:
        ctypes.CDLL("libc.so.6")
        # posix system
        return True
    except FileNotFoundError:
        # windows system
        return False


if __check_posix():
    from .posix.mmap import free_memory
else:
    from .win.mmap import free_memory
