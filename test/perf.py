import sys
import random
import itertools
from time import perf_counter
import gc
from typing import Callable, Any

import numpy as np
import torch

from ckpt_streamer import stream
from utils import temp_file


def perf(n: int, fn: Callable[[dict[str, Any]], None]):
    # N = 1024
    N = n // 1024
    WARM_UP = 1
    RUN = 5

    with temp_file() as f:
        with open(f, "wb") as f2:
            torch.save({f"key{i}": torch.zeros(n // N, dtype=torch.float32) for i in range(N)}, f2)

        # warm-up
        for _ in range(WARM_UP):
            sd = torch.load(f, map_location="cpu", weights_only=True, mmap=True)
            fn(sd)

        gc.collect()
        gc.collect()
        gc.disable()

        # run
        results = []
        for _ in range(RUN):
            sd = torch.load(f, map_location="cpu", weights_only=True, mmap=True)
            t0 = perf_counter()
            fn(sd)
            t1 = perf_counter()
            gc.collect()
            gc.collect()
            results.append(t1 - t0)
        gc.enable()

    return results


def test1(sd: dict[str, Any]):
    for k, v in sd.items():
        v.to(torch.bfloat16)


def test2(sd: dict[str, Any]):
    for _, k, v in stream(sd, memory_limit_mb=1024):
        v.to(torch.bfloat16)


def main():
    tests = [
        ("w/o ckpt_streamer", test1),
        ("w/ ckpt_streamer", test2),
    ]

    ns = [
        1 << 20,  # 1 MiB
        1 << 22,  # 4 MiB
        1 << 24,  # 16 MiB
        1 << 26,  # 64 MiB
        1 << 28,  # 256 MiB
        # 1 << 30,  # 1 GiB
        # 1 << 32,  # 4 GiB
    ]

    all_tests = list(itertools.product(tests, ns))
    random.shuffle(all_tests)

    for (name, fn), n in all_tests:
        print(name, n, file=sys.stderr)
        results = perf(n, fn)
        for r in results:
            print(f"{name},{n},{r}")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
