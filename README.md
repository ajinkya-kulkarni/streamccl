# streamccl

`streamccl` is a storage-first connected-component labeling (CCL) library for large 2D and 3D binary arrays.

It labels one chunk at a time, persists provisional labels, records only cross-chunk equivalences, resolves those equivalences through a disk-backed SQLite union-find, and then streams a final relabeling pass. The goal is to avoid materializing either the full image or a global connectivity graph in memory.

## Why

In-memory CCL is already solved well by SciPy, scikit-image, cc3d, and similar libraries. The problem addressed here is different: labeling arrays that are too large to treat as a single in-memory object while still producing globally correct component IDs across chunk boundaries.

## Install

```bash
pip install -e .
```

For Zarr support/tests:

```bash
pip install -e '.[zarr]'
```

## Usage

```python
import zarr
from streamccl import label

source = zarr.open_array("binary.zarr", mode="r")
out = zarr.open_array(
    "labels.zarr",
    mode="w",
    shape=source.shape,
    chunks=source.chunks,
    dtype="i8",
)

result = label(
    source,
    out,
    chunks=source.chunks,
    connectivity=3,      # 3D: 1=6-neighbor, 2=18-neighbor, 3=26-neighbor
    memory_limit="256MiB",
)

print(result.num_components)
```

NumPy arrays, `numpy.memmap`, Zarr arrays, HDF5 datasets, and other objects with `shape`, `dtype`, and NumPy-style slice reads/writes can be used as storage backends.

## Algorithm

1. **Local CCL** — read one chunk and run `scipy.ndimage.label` locally.
2. **Stable provisional IDs** — each local component receives the one-based global C-order index of its first voxel.
3. **Boundary reconciliation** — inspect only cross-chunk face/edge/corner contacts required by the requested connectivity.
4. **Disk-backed equivalence resolution** — merge provisional IDs in SQLite rather than constructing a global Python graph.
5. **Streaming canonicalization** — revisit chunks and replace merged IDs with their canonical global ID.

The resulting IDs are deterministic across chunk shapes, but intentionally not dense.

## Correctness

The test suite compares complete output arrays against `scipy.ndimage.label` for randomized 2D and 3D inputs, all supported connectivities, multiple chunk layouts, cross-boundary diagonal contacts, snaking components, memmap storage, and Zarr storage.

```bash
pytest
```

GitHub Actions tests Python 3.10 through 3.13 and covers both Zarr 2.x and Zarr 3.x.

## Benchmark

A local Linux benchmark used a chunked HDF5 dataset purely as a generic on-disk array backend (`streamccl` itself has no HDF5 dependency):

| Shape | Logical input + output | Chunks | Foreground density | Time | Throughput | Peak RSS |
|---|---:|---:|---:|---:|---:|---:|
| 512³ | 1.125 GiB | 64³ | 0.2% | 12.10 s | 11.09 Mvox/s | 170 MiB |
| 640³ | 2.197 GiB | 64³ | 0.1% | 23.35 s | 11.23 Mvox/s | 198 MiB |

Run it yourself:

```bash
pip install psutil h5py
PYTHONPATH=src python benchmarks/benchmark_hdf5.py \
  --shape 640 640 640 \
  --chunks 64 64 64 \
  --density 0.001 \
  --memory-limit 64MiB
```

These numbers are development measurements from one container, not a performance claim across machines. The benchmark is intended to demonstrate that resident memory can remain far below logical dataset size when using genuinely chunked storage.

`memory_limit` currently controls conservative chunk planning; it is **not** a hard operating-system RSS limit. With memory-mapped files, RSS can include reclaimable file-backed pages and can therefore be much higher than the planner budget.

## Scope of v1

- binary 2D and 3D arrays
- configurable connectivity
- deterministic global IDs across chunk shapes
- caller-provided output storage
- bounded-size chunk processing
- disk-backed equivalence state
- NumPy/memmap/Zarr-style sliceable stores

Not in v1: distributed scheduling, GPU kernels, resumability, dense output IDs, or microscopy-specific metadata.

## Release status

`0.1.0` is the first release-ready scope. The release gate is:

- complete output equality with SciPy reference labeling across randomized 2D/3D tests
- all supported connectivities and uneven chunk layouts
- deterministic IDs independent of chunk shape
- NumPy and memmap coverage
- Zarr 2.x and Zarr 3.x CI coverage
- adversarial cross-chunk component stress coverage
- disk-backed global equivalence state with bounded SQLite cache
- a reproducible million-node resolver benchmark

The SQLite database is ephemeral scratch state. `streamccl` disables SQLite journaling/fsync and uses an exclusive connection for this private database; interrupted runs are not resumable and output contents are undefined on failure. Put `workdir` on fast local scratch storage when possible.

## Adversarial boundary stress test

Sparse random volumes are not the hardest case for an out-of-core CCL resolver.
`benchmarks/benchmark_adversarial.py` creates independent one-voxel-wide filaments
that run along axis 0. Each filament crosses every chunk boundary on that axis,
forcing many distinct provisional-ID equivalences without collapsing into one
giant component.

Final `0.1.0` development measurements with `32³` chunks, 6-connectivity and a
`64MiB` planning budget:

| Shape | Independent components | Cross-chunk equivalences | Time | Throughput | Peak RSS |
|---|---:|---:|---:|---:|---:|
| 256³ | 16,384 | 114,688 | 3.90 s | 4.31 Mvox/s | 140 MiB |
| 384³ | 36,864 | 405,504 | 12.46 s | 4.54 Mvox/s | 153 MiB |
| 512³ | 65,536 | 983,040 | 31.74 s | 4.23 Mvox/s | 166 MiB |

Run it yourself:

```bash
PYTHONPATH=src python benchmarks/benchmark_adversarial.py \
  --shape 384 384 384 \
  --chunks 32 32 32 \
  --spacing 2 \
  --memory-limit 64MiB \
  --workdir /path/to/local/scratch
```

The resolver-only benchmark explicitly exercises the million-node regime:

```bash
PYTHONPATH=src python benchmarks/benchmark_resolver.py \
  --components 65536 \
  --segments 16 \
  --workdir /path/to/local/scratch
```

On the same development environment this produced **1,048,576 explicit nodes**,
**983,040 merges**, **12.21 s** union time and **0.159 s** finalization time
(**12.37 s total**). This benchmark isolates global equivalence resolution from
storage-backend throughput.

These are development measurements from one Linux container, not cross-machine
performance guarantees. The 512³ filament case demonstrates that the previous million-equivalence
scaling wall is gone in the current implementation. End-to-end performance is
still storage-sensitive, so users should benchmark their actual
Zarr/HDF5/object-store backend and place resolver scratch on local storage where
possible.

## Non-goals for 0.1

Distributed scheduling, GPU kernels, resumability, dense output IDs, arbitrary
label-value preservation, and microscopy-specific metadata are intentionally out
of scope. The library is a focused connected-component primitive, not a workflow
engine.
