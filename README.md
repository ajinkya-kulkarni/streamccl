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

GitHub Actions tests the Zarr round-trip against both Zarr 2.x and Zarr 3.x.

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

## Adversarial boundary stress test

Sparse random volumes are not the hardest case for an out-of-core CCL resolver. `benchmarks/benchmark_adversarial.py` creates thousands of independent one-voxel-wide filaments that run along axis 0. Each filament crosses every chunk boundary on that axis, forcing many distinct provisional-ID equivalences without collapsing them into one giant component.

Current development measurements with `32³` chunks and 6-connectivity:

| Shape | Independent components | Cross-chunk equivalences | Time | Throughput | Peak RSS |
|---|---:|---:|---:|---:|---:|
| 256³ | 16,384 | 114,688 | 6.21 s | 2.70 Mvox/s | 136 MiB |
| 384³ | 36,864 | 405,504 | 19.94 s | 2.84 Mvox/s | 145 MiB |

The resolver now batches each bounded boundary block into batched SQLite reads and upserts rather than issuing several SQL statements per equivalence. On the same development environment this reduced the 256³ stress case from 8.30 s to 6.21 s and the 384³ case from 25.64 s to 19.94 s, while keeping peak RSS effectively flat.

A 512³ version of this deliberately pathological workload creates about 983,000 cross-chunk equivalences and did not complete within a 120-second stress-run cap in the current container. That is an explicit v1 scaling limit, not a claimed success. The next optimization target is canonicalization/resolver I/O near the million-node regime.

Run the stress case with:

```bash
PYTHONPATH=src python benchmarks/benchmark_adversarial.py \
  --shape 384 384 384 \
  --chunks 32 32 32 \
  --spacing 2 \
  --memory-limit 64MiB
```

## Scope of v1

- binary 2D and 3D arrays
- configurable connectivity
- deterministic global IDs across chunk shapes
- caller-provided output storage
- bounded-size chunk processing
- disk-backed equivalence state
- NumPy/memmap/Zarr-style sliceable stores

Not in v1: distributed scheduling, GPU kernels, resumability, dense output IDs, or microscopy-specific metadata.

## Development status

This is an early prototype. The current design prioritizes correctness and bounded global bookkeeping over peak single-machine speed. Before a stable release, the main remaining work is failure/recovery semantics and improving resolver/canonicalization scaling near the million-equivalence regime.
