"""Storage-first connected-component labeling for 2D and 3D arrays."""

from __future__ import annotations

import itertools
import math
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np
from scipy import ndimage as ndi

from ._equivalence import Equivalences

_MAX_ID = np.iinfo(np.int64).max
_PLANNER_BYTES_PER_VOXEL = 128
_RESOLVER_CACHE = 8 * 1024 * 1024


@dataclass(frozen=True)
class LabelResult:
    """The destination and the number of global foreground components.

    Label IDs are the one-based C-order index of each component's first voxel.
    They are deterministic across chunk shapes but are not necessarily dense.
    """

    labels: Any
    num_components: int


def _bytes(value: int | str) -> int:
    if isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_)):
        result = int(value)
    elif isinstance(value, str):
        match = re.fullmatch(r"\s*(\d+)\s*(B|KiB|MiB|GiB|TiB)\s*", value, re.I)
        if match is None:
            raise ValueError("memory_limit must be bytes or a value such as '256MiB'")
        scale = {"b": 1, "kib": 1024, "mib": 1024**2, "gib": 1024**3, "tib": 1024**4}
        result = int(match[1]) * scale[match[2].lower()]
    else:
        raise TypeError("memory_limit must be an integer number of bytes or a size string")
    if result <= _RESOLVER_CACHE + _PLANNER_BYTES_PER_VOXEL:
        raise ValueError("memory_limit is too small for the resolver and one voxel")
    return result


def _plan(shape: tuple[int, ...], chunks: Sequence[int] | None, budget: int,
          source_chunks: Any = None) -> tuple[int, ...]:
    if chunks is None:
        if (isinstance(source_chunks, (tuple, list)) and len(source_chunks) == len(shape)
                and all(isinstance(x, (int, np.integer)) for x in source_chunks)):
            proposed = tuple(int(x) for x in source_chunks)
        else:
            proposed = tuple(min(s, 256 if len(shape) == 2 else 64) for s in shape)
        proposed = tuple(max(1, min(s, c)) for s, c in zip(shape, proposed))
        while math.prod(proposed) * _PLANNER_BYTES_PER_VOXEL + _RESOLVER_CACHE > budget:
            axis = max(range(len(shape)), key=lambda i: proposed[i])
            if proposed[axis] == 1:
                raise ValueError("memory_limit is too small")
            temp = list(proposed)
            temp[axis] = max(1, temp[axis] // 2)
            proposed = tuple(temp)
        return proposed
    if len(chunks) != len(shape) or any(
        not isinstance(c, (int, np.integer)) or isinstance(c, (bool, np.bool_)) or c <= 0
        for c in chunks
    ):
        raise ValueError("chunks must contain one positive integer per axis")
    proposed = tuple(int(c) for c in chunks)
    effective = tuple(min(s, c) for s, c in zip(shape, proposed))
    if math.prod(effective) * _PLANNER_BYTES_PER_VOXEL + _RESOLVER_CACHE > budget:
        raise ValueError("chunks exceed the conservative memory planning budget")
    return proposed


def _grid(shape: tuple[int, ...], chunks: tuple[int, ...]) -> tuple[int, ...]:
    return tuple((s + c - 1) // c for s, c in zip(shape, chunks))


def _bounds(index: tuple[int, ...], shape: tuple[int, ...],
            chunks: tuple[int, ...]) -> tuple[tuple[int, int], ...]:
    return tuple((i * c, min((i + 1) * c, s)) for i, c, s in zip(index, chunks, shape))


def _slices(bounds: tuple[tuple[int, int], ...]) -> tuple[slice, ...]:
    return tuple(slice(a, b) for a, b in bounds)


def _indices(grid: tuple[int, ...]) -> Iterator[tuple[int, ...]]:
    return np.ndindex(*grid)


def _local_ids(binary: np.ndarray, bounds: tuple[tuple[int, int], ...],
               shape: tuple[int, ...], structure: np.ndarray) -> tuple[np.ndarray, int]:
    # SciPy's labeling is local only.  Stable global keys are derived from each
    # local component's first voxel, not from the chunk's position or label order.
    local, count = ndi.label(binary, structure=structure)
    if count == 0:
        return np.zeros(local.shape, dtype=np.int64), 0
    flat = local.ravel()
    first = np.full(count + 1, flat.size, dtype=np.int64)
    np.minimum.at(first, flat, np.arange(flat.size, dtype=np.int64))
    coords = np.unravel_index(first[1:], local.shape)
    global_coords = tuple(c + lo for c, (lo, _) in zip(coords, bounds))
    keys = np.empty(count + 1, dtype=np.int64)
    keys[0] = 0
    keys[1:] = np.ravel_multi_index(global_coords, shape) + 1
    return keys[local], int(count)


def _offsets(ndim: int, connectivity: int) -> tuple[tuple[int, ...], ...]:
    return tuple(
        offset for offset in itertools.product((-1, 0, 1), repeat=ndim)
        if 0 < sum(v != 0 for v in offset) <= connectivity
    )


def _boundary_pairs(out: Any, shape: tuple[int, ...], chunks: tuple[int, ...],
                    connectivity: int) -> Iterator[np.ndarray]:
    """Yield deduplicated equivalences for one boundary comparison at a time."""
    ndim = len(shape)
    grid = _grid(shape, chunks)
    offsets = _offsets(ndim, connectivity)
    # A pair of chunks is visited once.  All relevant voxel offsets are
    # considered, including face/edge/corner contacts at 8/18/26 connectivity.
    neighbor_deltas = tuple(
        d for d in itertools.product((-1, 0, 1), repeat=ndim)
        if any(d) and next(v for v in d if v != 0) < 0
        and sum(v != 0 for v in d) <= connectivity
    )
    for index in _indices(grid):
        current = _bounds(index, shape, chunks)
        for delta in neighbor_deltas:
            other_index = tuple(i + d for i, d in zip(index, delta))
            if any(i < 0 or i >= g for i, g in zip(other_index, grid)):
                continue
            other = _bounds(other_index, shape, chunks)
            for offset in offsets:
                if any(d != 0 and v != d for d, v in zip(delta, offset)):
                    continue
                # A point x in the current chunk is adjacent to x+offset in
                # the other chunk.  Intersect the two global coordinate ranges.
                starts = tuple(max(a, c - v) for (a, _), (c, _), v in zip(current, other, offset))
                stops = tuple(min(b, d - v) for (_, b), (_, d), v in zip(current, other, offset))
                if any(a >= b for a, b in zip(starts, stops)):
                    continue
                left = np.asarray(out[tuple(slice(a, b) for a, b in zip(starts, stops))])
                right = np.asarray(out[tuple(slice(a + v, b + v) for a, b, v in zip(starts, stops, offset))])
                mask = (left != 0) & (right != 0) & (left != right)
                if not np.any(mask):
                    continue
                pairs = np.stack((left[mask], right[mask]), axis=1)
                pairs.sort(axis=1)
                yield np.unique(pairs, axis=0)


def _replace(labels: np.ndarray, old: np.ndarray, new: np.ndarray) -> np.ndarray:
    if not len(old):
        return labels
    # Background is always zero and never participates in equivalence
    # resolution.  Searching only foreground values avoids doing a binary
    # search for every background voxel, which matters for the sparse masks
    # streamccl is primarily designed to process.
    foreground = labels != 0
    values = labels[foreground]
    positions = np.searchsorted(old, values)
    clipped = np.minimum(positions, len(old) - 1)
    hit = (positions < len(old)) & (old[clipped] == values)
    values[hit] = new[positions[hit]]
    labels[foreground] = values
    return labels


def label(source: Any, out: Any, *, chunks: Sequence[int] | None = None,
          connectivity: int | None = None, memory_limit: int | str = "256MiB",
          workdir: str | Path | None = None) -> LabelResult:
    """Label a binary 2D/3D image without materializing the entire image.

    Nonzero values are foreground. `source` and `out` must be distinct arrays
    supporting shape, dtype and NumPy-style slice reads/writes (NumPy, memmap,
    and Zarr). `out` is required, must be writable, and is overwritten in place.
    On failure its contents are undefined; v1 does not support resumability.

    `memory_limit` is a conservative chunk-planning budget, not an operating-
    system RSS guarantee.  The SQLite resolver uses a bounded page cache; its
    temporary database may grow with the number of boundary components.
    Labels are stable, non-dense, one-based global C-order first-voxel indices.
    """
    shape = tuple(int(s) for s in source.shape)
    if len(shape) not in (2, 3) or any(s < 0 for s in shape):
        raise ValueError("source must be a 2D or 3D array")
    if tuple(out.shape) != shape:
        raise ValueError("source and out must have the same shape")
    if source is out:
        raise ValueError("source and out must be distinct arrays")
    if math.prod(shape) > _MAX_ID:
        raise ValueError("image is too large for int64 component IDs")
    dtype = np.dtype(out.dtype)
    if dtype.kind not in "iu" or int(np.iinfo(dtype).max) < math.prod(shape):
        raise ValueError("out must have an integer dtype capable of holding global voxel IDs")
    if isinstance(source, np.ndarray) and isinstance(out, np.ndarray) and np.may_share_memory(source, out):
        raise ValueError("source and out must not share memory")
    if connectivity is None:
        connectivity = 1
    if not isinstance(connectivity, (int, np.integer)) or isinstance(connectivity, (bool, np.bool_)) or not 1 <= connectivity <= len(shape):
        raise ValueError("connectivity must be an integer between 1 and source.ndim")
    connectivity = int(connectivity)
    budget = _bytes(memory_limit)
    tile = _plan(shape, chunks, budget, getattr(source, "chunks", None))
    if not math.prod(shape):
        return LabelResult(out, 0)
    structure = ndi.generate_binary_structure(len(shape), connectivity)
    grid = _grid(shape, tile)
    total = 0
    with tempfile.TemporaryDirectory(prefix="streamccl-", dir=workdir) as directory:
        resolver = Equivalences(Path(directory) / "equivalences.sqlite")
        try:
            # Pass 1: local CCL, immediately persisted as stable provisional IDs.
            for index in _indices(grid):
                bounds = _bounds(index, shape, tile)
                sl = _slices(bounds)
                binary = np.asarray(source[sl]) != 0
                provisional, count = _local_ids(binary, bounds, shape, structure)
                out[sl] = provisional
                total += count
            # Pass 2: visit every cross-chunk neighbor pair; never build a
            # global graph or materialize the full image.
            for pairs in _boundary_pairs(out, shape, tile, connectivity):
                resolver.union_many(pairs)
            resolver.flush()
            count = total - resolver.merges
            # Pass 3: apply global canonical IDs.  If the complete sparse
            # replacement map fits in the unused planning budget, materialize
            # it once; otherwise keep canonical lookups disk-backed per chunk.
            if resolver.merges:
                effective = tuple(min(s, c) for s, c in zip(shape, tile))
                planned_chunk_bytes = math.prod(effective) * _PLANNER_BYTES_PER_VOXEL
                spare = max(0, budget - planned_chunk_bytes - _RESOLVER_CACHE)
                global_map = resolver.materialize_replacements(spare)
                for index in _indices(grid):
                    sl = _slices(_bounds(index, shape, tile))
                    block = np.asarray(out[sl]).astype(np.int64, copy=True)
                    if global_map is None:
                        old, new = resolver.replacements(np.unique(block))
                    else:
                        old, new = global_map
                        # Provisional IDs are first-voxel indices created
                        # inside this chunk.  Restrict the global sparse map
                        # to the numeric range actually present in the block
                        # before searchsorted-based replacement.
                        foreground = block != 0
                        if np.any(foreground):
                            lo = int(block[foreground].min())
                            hi = int(block[foreground].max())
                            start = int(np.searchsorted(old, lo, side="left"))
                            stop = int(np.searchsorted(old, hi, side="right"))
                            old = old[start:stop]
                            new = new[start:stop]
                        else:
                            old = old[:0]
                            new = new[:0]
                    if len(old):
                        out[sl] = _replace(block, old, new)
            return LabelResult(out, count)
        finally:
            resolver.close()
