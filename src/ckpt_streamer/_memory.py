"""
for development purpose
"""

from threading import Thread
from multiprocessing import Queue
from queue import Empty as QueueEmpty, Full as QueueFull
import time
from contextlib import contextmanager

import psutil


def _show_rss(q: Queue, interval=1, base_rss=0):
    p = psutil.Process()
    last = time.time()
    while True:
        try:
            q.get_nowait()
            # completed
            break
        except QueueEmpty:
            pass
        rss = p.memory_info().rss
        t1 = time.time()
        if t1 - last > interval:
            print(f"RSS: {t1}, {(rss - base_rss) / 1024:.0f} KiB")
            last = time.time()
        time.sleep(0.01)


@contextmanager
def show_rss(interval: float = 1.0, base_rss: int = 0, timeout: float = 1.0):
    q = Queue(1)
    th = Thread(target=_show_rss, args=(q, interval, base_rss))
    th.start()

    try:
        yield
    finally:
        try:
            q.put_nowait(None)
            th.join(timeout)
        except QueueFull:
            pass


def current_rss():
    return psutil.Process().memory_info().rss
