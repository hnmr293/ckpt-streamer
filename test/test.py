import os
import tempfile
from contextlib import contextmanager
import gc
import time

import torch

from ckpt_streamer import stream, _memory


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


@contextmanager
def run_test():
    size = 100 * MiB
    N = 128

    with temp_file() as f_:
        with open(f_, "wb") as f:
            torch.save({f"key{i}": torch.zeros(size // 4 // N, dtype=torch.float32) for i in range(N)}, f)

        gc.collect()

        # rss0 = _memory.current_rss()
        rss0 = 0
        with _memory.show_rss(0.1, rss0):
            sd = torch.load(f_, map_location="cpu", weights_only=True, mmap=True)
            yield sd


def test1():
    with run_test() as sd:
        for k, v in sd.items():
            v.to(torch.bfloat16)
            time.sleep(0.1)


def test2():
    with run_test() as sd:
        for _, k, v in stream(sd, memory_limit_mb=10):
            v.to(torch.bfloat16)
            time.sleep(0.1)


if __name__ == "__main__":
    import sys

    if len(sys.argv) == 2:
        if sys.argv[1] == "test1":
            test1()
        elif sys.argv[1] == "test2":
            test2()
        else:
            raise RuntimeError(f"Unknown test: {sys.argv[1]}")
    else:
        import subprocess

        a1 = subprocess.run([sys.executable, __file__, "test1"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        a2 = subprocess.run([sys.executable, __file__, "test2"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        a1.check_returncode()
        a2.check_returncode()

        a1 = [line for line in a1.stdout.decode().strip().split("\n") if line.startswith("RSS: ")]
        a2 = [line for line in a2.stdout.decode().strip().split("\n") if line.startswith("RSS: ")]

        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        for i, lines in enumerate([a1, a2], start=1):
            xs = []
            ys = []
            for line in lines:
                line = line.lstrip("RSS: ")
                t, v = line.split(",")
                t = float(t.strip())
                v = float(v.strip().split()[0])
                xs.append(t)
                ys.append(v)
            xs = [x - xs[0] for x in xs]
            ys = [(y - ys[0]) / 1024 for y in ys]
            ax.plot(xs, ys, label=f"test{i}")
        ax.set_xlabel("time (s)")
        ax.set_ylabel("Process RSS (MiB)")
        ax.legend()
        fig.savefig(os.path.dirname(__file__) + "/test.png")
