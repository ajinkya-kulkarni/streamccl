"""Bounded-memory benchmark using a chunked HDF5 dataset as generic storage.

HDF5 is used only as a local benchmark backend because it is available in the
benchmark environment. streamccl itself does not depend on h5py.
"""
from __future__ import annotations

import argparse
import json
import tempfile
import threading
import time
from pathlib import Path

import h5py
import numpy as np
import psutil

from streamccl import label


class MemorySampler:
    def __init__(self, interval: float = 0.02) -> None:
        self.interval = interval
        self.process = psutil.Process()
        self.peak_rss = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _sample(self) -> None:
        self.peak_rss = max(self.peak_rss, int(self.process.memory_info().rss))

    def _run(self) -> None:
        while not self._stop.wait(self.interval):
            self._sample()

    def __enter__(self) -> "MemorySampler":
        self._sample()
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self._stop.set()
        self._thread.join()
        self._sample()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--shape", nargs="+", type=int, default=[512, 512, 512])
    p.add_argument("--chunks", nargs="+", type=int, default=[64, 64, 64])
    p.add_argument("--density", type=float, default=0.002)
    p.add_argument("--connectivity", type=int, default=1)
    p.add_argument("--memory-limit", default="64MiB")
    p.add_argument("--seed", type=int, default=7)
    args = p.parse_args()

    shape = tuple(args.shape)
    chunks = tuple(args.chunks)
    voxels = int(np.prod(shape, dtype=np.int64))

    with tempfile.TemporaryDirectory(prefix="streamccl-h5bench-") as td:
        path = Path(td) / "bench.h5"
        with h5py.File(path, "w", rdcc_nbytes=8 * 1024**2) as f:
            source = f.create_dataset("source", shape=shape, chunks=chunks, dtype="u1")
            out = f.create_dataset("labels", shape=shape, chunks=chunks, dtype="i8")
            rng = np.random.default_rng(args.seed)
            slab = chunks[0]
            foreground = 0
            for start in range(0, shape[0], slab):
                stop = min(start + slab, shape[0])
                block = rng.random((stop - start, *shape[1:])) < args.density
                foreground += int(block.sum())
                source[start:stop] = block
            f.flush()

            with MemorySampler() as memory:
                started = time.perf_counter()
                result = label(
                    source,
                    out,
                    chunks=chunks,
                    connectivity=args.connectivity,
                    memory_limit=args.memory_limit,
                    workdir=td,
                )
                f.flush()
                elapsed = time.perf_counter() - started

            payload = {
                "shape": shape,
                "chunks": chunks,
                "density": args.density,
                "logical_input_bytes": voxels,
                "logical_output_bytes": voxels * 8,
                "logical_total_bytes": voxels * 9,
                "elapsed_seconds": elapsed,
                "throughput_mvox_s": voxels / elapsed / 1e6,
                "peak_rss_bytes": memory.peak_rss,
                "peak_rss_mib": memory.peak_rss / 1024**2,
                "components": result.num_components,
                "foreground_voxels": foreground,
            }
            print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
