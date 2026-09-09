"""Adversarial boundary-equivalence benchmark for streamccl.

The ``filaments`` pattern creates many independent one-voxel-wide components
that run along axis 0. Every filament crosses every chunk boundary on that
axis, forcing the resolver to reconcile a large number of distinct provisional
component IDs while keeping the global components mutually disconnected.

This targets the global-equivalence stage rather than local CCL complexity.
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


def _write_filaments(source: h5py.Dataset, spacing: int) -> int:
    if source.ndim != 3:
        raise ValueError("filaments benchmark currently requires a 3D shape")
    if spacing < 2:
        raise ValueError("spacing must be at least 2 for disjoint 6-neighbor filaments")
    plane = np.zeros(source.shape[1:], dtype=np.uint8)
    plane[::spacing, ::spacing] = 1
    count = int(plane.sum())
    slab = max(1, int(source.chunks[0]))
    for start in range(0, source.shape[0], slab):
        stop = min(start + slab, source.shape[0])
        source[start:stop] = np.broadcast_to(plane, (stop - start, *plane.shape))
    return count


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--shape", nargs=3, type=int, default=[384, 384, 384])
    p.add_argument("--chunks", nargs=3, type=int, default=[32, 32, 32])
    p.add_argument("--spacing", type=int, default=2)
    p.add_argument("--memory-limit", default="64MiB")
    args = p.parse_args()

    shape = tuple(args.shape)
    chunks = tuple(args.chunks)
    voxels = int(np.prod(shape, dtype=np.int64))
    axis0_chunks = (shape[0] + chunks[0] - 1) // chunks[0]

    with tempfile.TemporaryDirectory(prefix="streamccl-adversarial-") as td:
        path = Path(td) / "bench.h5"
        with h5py.File(path, "w", rdcc_nbytes=8 * 1024**2) as f:
            source = f.create_dataset("source", shape=shape, chunks=chunks, dtype="u1")
            out = f.create_dataset("labels", shape=shape, chunks=chunks, dtype="i8")
            expected_components = _write_filaments(source, args.spacing)
            foreground = expected_components * shape[0]
            expected_axis0_equivalences = expected_components * max(0, axis0_chunks - 1)
            f.flush()

            with MemorySampler() as memory:
                started = time.perf_counter()
                result = label(
                    source,
                    out,
                    chunks=chunks,
                    connectivity=1,
                    memory_limit=args.memory_limit,
                    workdir=td,
                )
                f.flush()
                elapsed = time.perf_counter() - started

            if result.num_components != expected_components:
                raise RuntimeError(
                    f"expected {expected_components} components, got {result.num_components}"
                )

            payload = {
                "pattern": "axis0_filaments",
                "shape": shape,
                "chunks": chunks,
                "spacing": args.spacing,
                "logical_input_bytes": voxels,
                "logical_output_bytes": voxels * 8,
                "logical_total_bytes": voxels * 9,
                "foreground_voxels": foreground,
                "expected_components": expected_components,
                "expected_axis0_equivalences": expected_axis0_equivalences,
                "elapsed_seconds": elapsed,
                "throughput_mvox_s": voxels / elapsed / 1e6,
                "equivalence_rate_k_s": expected_axis0_equivalences / elapsed / 1e3,
                "peak_rss_bytes": memory.peak_rss,
                "peak_rss_mib": memory.peak_rss / 1024**2,
            }
            print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
