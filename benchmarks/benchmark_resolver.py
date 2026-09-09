"""Benchmark only the disk-backed global-equivalence resolver.

This removes image-storage throughput from the measurement and targets the
path that previously became pathological around one million provisional
component IDs. The default workload models 65,536 independent components
continued through 16 chunks: 1,048,576 explicit nodes and 983,040 merges.
"""
from __future__ import annotations

import argparse
import json
import tempfile
import time
from pathlib import Path

import numpy as np

from streamccl._equivalence import Equivalences


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--components", type=int, default=65_536)
    parser.add_argument("--segments", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--workdir", type=Path, default=None)
    args = parser.parse_args()
    if args.components <= 0 or args.segments <= 0 or args.batch_size <= 0:
        raise ValueError("components, segments, and batch-size must be positive")

    with tempfile.TemporaryDirectory(prefix="streamccl-resolver-", dir=args.workdir) as td:
        resolver = Equivalences(Path(td) / "equivalences.sqlite")
        union_started = time.perf_counter()
        try:
            base = np.arange(args.components, dtype=np.int64) + 1
            for segment in range(1, args.segments):
                left = base + (segment - 1) * args.components
                right = base + segment * args.components
                for start in range(0, args.components, args.batch_size):
                    stop = min(start + args.batch_size, args.components)
                    resolver.union_many(np.column_stack((left[start:stop], right[start:stop])))
            resolver.flush()
            union_seconds = time.perf_counter() - union_started

            finalize_started = time.perf_counter()
            resolver.finalize()
            finalize_seconds = time.perf_counter() - finalize_started

            node_count = resolver.db.execute("SELECT count(*) FROM nodes").fetchone()[0]
            depth_gt_one = resolver.db.execute(
                "SELECT count(*) FROM nodes AS c JOIN nodes AS p ON p.id = c.parent "
                "WHERE c.parent != c.id AND p.parent != p.id"
            ).fetchone()[0]
            expected_nodes = args.components * args.segments
            expected_merges = args.components * (args.segments - 1)
            if node_count != expected_nodes or resolver.merges != expected_merges:
                raise RuntimeError(
                    f"unexpected resolver state: nodes={node_count}, merges={resolver.merges}"
                )
            if depth_gt_one:
                raise RuntimeError(f"finalize left {depth_gt_one} non-flat nodes")

            payload = {
                "components": args.components,
                "segments": args.segments,
                "nodes": node_count,
                "merges": resolver.merges,
                "union_seconds": union_seconds,
                "finalize_seconds": finalize_seconds,
                "total_seconds": union_seconds + finalize_seconds,
                "merge_rate_k_s": expected_merges / union_seconds / 1e3,
            }
            print(json.dumps(payload, indent=2))
        finally:
            resolver.close()


if __name__ == "__main__":
    main()
