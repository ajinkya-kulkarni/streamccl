import itertools
import math

import numpy as np
import pytest
from scipy import ndimage as ndi

from streamccl import label
from streamccl._core import _boundary_pairs, _bytes, _plan
from streamccl._equivalence import Equivalences


def reference(image, connectivity):
    labels, count = ndi.label(image, ndi.generate_binary_structure(image.ndim, connectivity))
    expected = np.zeros(image.shape, dtype=np.int64)
    if count:
        flat = labels.ravel()
        first = np.full(count + 1, flat.size, dtype=np.int64)
        np.minimum.at(first, flat, np.arange(flat.size))
        first[0] = -1
        expected = (first[labels] + 1).astype(np.int64)
    return expected, count


def check(image, chunks, connectivity, *, destination=None, **kwargs):
    out = np.empty(image.shape, dtype=np.int64) if destination is None else destination
    result = label(image, out, chunks=chunks, connectivity=connectivity, **kwargs)
    expected, count = reference(np.asarray(image), connectivity)
    np.testing.assert_array_equal(out, expected)
    assert result.labels is out
    assert result.num_components == count
    return result


@pytest.mark.parametrize("ndim", [2, 3])
def test_random_partitions(ndim):
    rng = np.random.default_rng(148)
    for connectivity in range(1, ndim + 1):
        for shape in (tuple([2] * ndim), tuple([7] * ndim)):
            for density in (0.0, 0.05, 0.45, 0.95, 1.0):
                image = rng.random(shape) < density
                for chunks in (tuple([1] * ndim), tuple([3] * ndim), shape):
                    check(image, chunks, connectivity)


@pytest.mark.parametrize("ndim", [2, 3])
def test_single_voxel_and_diagonal_contacts(ndim):
    shape = (4,) * ndim
    image = np.zeros(shape, bool)
    image[(1,) * ndim] = True
    image[(2,) * ndim] = True
    for connectivity in range(1, ndim + 1):
        check(image, (2,) * ndim, connectivity)
    for offset in itertools.product((-1, 0, 1), repeat=ndim):
        if not any(offset):
            continue
        image = np.zeros(shape, bool)
        image[(1,) * ndim] = True
        image[tuple(1 + v for v in offset)] = True
        for connectivity in range(1, ndim + 1):
            check(image, (2,) * ndim, connectivity)


def test_snaking_component_and_chunk_independent_ids():
    image = np.zeros((11, 13, 15), bool)
    image[1, 1:12, 1] = True
    image[1:10, 11, 1] = True
    image[9, 1:12, 1] = True
    image[9, 1, 1:14] = True
    image[0, 0, 0] = True
    image[10, 12, 14] = True
    outputs = []
    for chunks in ((2, 3, 4), (5, 7, 6), (11, 13, 15), (1, 1, 1)):
        out = np.empty(image.shape, np.int64)
        check(image, chunks, 1, destination=out)
        outputs.append(out)
    for out in outputs[1:]:
        np.testing.assert_array_equal(out, outputs[0])


def test_nonzero_input_and_background():
    image = np.array([[0, -7, 0], [np.nan, 0, 3], [0, 0, 0]], dtype=float)
    check(image, (2, 2), 1)


def test_empty_arrays():
    for shape in ((0, 5), (3, 0, 4)):
        out = np.empty(shape, dtype=np.int64)
        result = label(np.empty(shape, bool), out)
        assert result.num_components == 0
        assert result.labels is out


def test_memmap_larger_than_chunk(tmp_path):
    shape = (35, 41, 29)
    source = np.memmap(tmp_path / "input.dat", dtype="uint8", mode="w+", shape=shape)
    out = np.memmap(tmp_path / "output.dat", dtype="int64", mode="w+", shape=shape)
    rng = np.random.default_rng(80)
    source[:] = rng.random(shape) < 0.12
    check(source, (7, 9, 5), 3, destination=out, workdir=tmp_path)
    assert not list(tmp_path.glob("streamccl-*"))
    assert np.count_nonzero(out) == np.count_nonzero(source)


def test_integer_output_and_invalid_input():
    image = np.eye(6, dtype=bool)
    check(image, (2, 3), 2, destination=np.empty(image.shape, np.uint16))
    with pytest.raises(ValueError, match="integer dtype"):
        label(np.ones((20, 20), bool), np.empty((20, 20), np.uint8))
    with pytest.raises(ValueError, match="same shape"):
        label(image, np.empty((6, 7), np.int64))
    with pytest.raises(ValueError, match="share memory"):
        shared = image.astype(np.int64)
        label(shared, shared)
    with pytest.raises(ValueError, match="connectivity"):
        label(image, np.empty(image.shape, np.int64), connectivity=3)
    with pytest.raises(ValueError, match="connectivity"):
        label(image, np.empty(image.shape, np.int64), connectivity=True)
    with pytest.raises(ValueError, match="chunks"):
        label(image, np.empty(image.shape, np.int64), chunks=(0, 2))
    with pytest.raises(ValueError, match="memory"):
        label(image, np.empty(image.shape, np.int64), memory_limit="not a size")
    with pytest.raises(ValueError, match="2D or 3D"):
        label(np.ones(5), np.empty(5, np.int64))


def test_memory_planner():
    budget = _bytes("16MiB")
    chunks = _plan((1000, 2000, 3000), None, budget)
    assert math.prod(chunks) * 128 + 8 * 1024**2 <= budget
    assert all(c > 0 for c in chunks)
    with pytest.raises(ValueError, match="budget"):
        _plan((1000, 1000), (1000, 1000), budget)
    assert _bytes("2GiB") == 2 * 1024**3
    assert _bytes(16_000_000) == 16_000_000


def test_no_global_reads():
    class Guarded:
        def __init__(self, array):
            self.array = array
            self.shape = array.shape
            self.dtype = array.dtype
            self.chunks = (4, 5)

        def __array__(self, dtype=None):
            return np.asarray(self.array, dtype=dtype)

        def __getitem__(self, key):
            assert all(isinstance(s, slice) and s.start is not None and s.stop is not None for s in key)
            assert math.prod(s.stop - s.start for s in key) <= 20
            return self.array[key]

        def __setitem__(self, key, value):
            assert math.prod(s.stop - s.start for s in key) <= 20
            self.array[key] = value

    source = np.eye(17, 19, dtype=bool)
    out = np.zeros(source.shape, np.int64)
    check(Guarded(source), (4, 5), 2, destination=Guarded(out))


def test_sqlite_resolver_implicit_singletons_and_transitive_merges(tmp_path):
    resolver = Equivalences(tmp_path / "eq.sqlite")
    try:
        for a, b in ((90, 20), (80, 30), (20, 30), (30, 10), (90, 10)):
            resolver.union(a, b)
        assert resolver.merges == 4
        assert resolver.find(90)[2] == 10
        assert resolver.find(1000) == (1000, 1, 1000)
        old, new = resolver.replacements(np.array([0, 10, 20, 30, 80, 90, 1000]))
        assert dict(zip(old.tolist(), new.tolist())) == {20: 10, 30: 10, 80: 10, 90: 10}
        resolver.flush()
        assert resolver.db.execute("SELECT count(*) FROM nodes").fetchone()[0] == 5
    finally:
        resolver.close()


def test_boundary_pairs_match_reference_edges():
    shape = (5, 6, 4)
    chunks = (2, 3, 2)
    out = np.arange(1, math.prod(shape) + 1, dtype=np.int64).reshape(shape)
    for connectivity in (1, 2, 3):
        actual = set()
        for pairs in _boundary_pairs(out, shape, chunks, connectivity):
            actual.update(map(tuple, pairs.tolist()))
        expected = set()
        for index in np.ndindex(*shape):
            for offset in itertools.product((-1, 0, 1), repeat=3):
                if not 0 < sum(v != 0 for v in offset) <= connectivity:
                    continue
                neighbor = tuple(a + b for a, b in zip(index, offset))
                if any(v < 0 or v >= s for v, s in zip(neighbor, shape)):
                    continue
                if tuple(a // c for a, c in zip(index, chunks)) == tuple(a // c for a, c in zip(neighbor, chunks)):
                    continue
                expected.add(tuple(sorted((int(out[index]), int(out[neighbor])))))
        assert actual == expected


def test_zarr_array(tmp_path):
    zarr = pytest.importorskip("zarr")
    image = np.random.default_rng(9).random((13, 15, 11)) < 0.2
    source = zarr.open_array(str(tmp_path / "input.zarr"), mode="w", shape=image.shape, chunks=(4, 5, 3), dtype="u1")
    out = zarr.open_array(str(tmp_path / "output.zarr"), mode="w", shape=image.shape, chunks=(4, 5, 3), dtype="i8")
    source[:] = image
    check(source, (4, 5, 3), 3, destination=out)
