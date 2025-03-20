import os
from contextlib import contextmanager
import tempfile

MiB = 1024**2
GiB = 1024**3
TiB = 1024**4


@contextmanager
def temp_file():
    fd, file = tempfile.mkstemp()
    with open(file, "wb") as f:
        pass
    try:
        yield file
    finally:
        os.close(fd)
        os.unlink(file)


@contextmanager
def huge_file(size=100 * GiB, flag="wb"):
    with temp_file() as file:
        os.truncate(file, size)
        with open(file, flag) as f:
            yield f
