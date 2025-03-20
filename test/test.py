import os
from contextlib import contextmanager
import gc
import time

import torch

from ckpt_streamer import stream, _memory
from utils import temp_file


@contextmanager
def run_test(interval=0.1):
    size = 4 * 1024 * 1024 * 1024  # 4 GiB
    N = 1024

    with temp_file() as f_:
        with open(f_, "wb") as f:
            torch.save({f"key{i}": torch.zeros(size // N, dtype=torch.float32) for i in range(N)}, f)

        gc.collect()

        with _memory.show_rss(interval):
            sd = torch.load(f_, map_location="cpu", weights_only=True, mmap=True)
            yield sd


def test1():
    with run_test() as sd:
        for k, v in sd.items():
            v.to(torch.bfloat16)
            time.sleep(0.01)


def test2():
    with run_test() as sd:
        for _, k, v in stream(sd, memory_limit_mb=1024):
            v.to(torch.bfloat16)
            time.sleep(0.01)


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
        from locale import getpreferredencoding

        encoding = getpreferredencoding()

        a1 = subprocess.run([sys.executable, __file__, "test1"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        a2 = subprocess.run([sys.executable, __file__, "test2"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        error = False
        if a1.returncode != 0:
            error = True
            print(a1.stderr.decode(encoding))
        if a2.returncode != 0:
            error = True
            print(a2.stderr.decode(encoding))

        if error:
            raise RuntimeError("Test failed")

        if len(a1.stderr) != 0:
            print("test1 stderr:")
            print(a1.stderr.decode(encoding))
        if len(a2.stderr) != 0:
            print("test2 stderr:")
            print(a2.stderr.decode(encoding))

        a1 = [line for line in a1.stdout.decode().strip().split("\n") if line.startswith("RSS: ")]
        a2 = [line for line in a2.stdout.decode().strip().split("\n") if line.startswith("RSS: ")]

        import plotly.graph_objects as go

        fig = go.Figure()

        for name, lines in [("w/o ckpt_streamer", a1), ("w/ ckpt_streamer", a2)]:
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
            ys = [y / 1024 for y in ys]
            fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines+markers", name=name))

        fig.update_layout(
            title="Physical Memory Usage",
            xaxis_title="time (s)",
            yaxis_title="Process RSS (MiB)",
        )

        fig.write_image(os.path.dirname(__file__) + "/test.png")
