import numpy as np
from scipy import ndimage as ndi

from streamccl import label
from streamccl._equivalence import Equivalences


def _reference(image: np.ndarray) -> tuple[np.ndarray, int]:
    labels, count = ndi.label(image, ndi.generate_binary_structure(image.ndim, 1))
    expected = np.zeros(image.shape, dtype=np.int64)
    if count:
        flat = labels.ravel()
        first = np.full(count + 1, flat.size, dtype=np.int64)
        np.minimum.at(first, flat, np.arange(flat.size))
        first[0] = -1
        expected = (first[labels] + 1).astype(np.int64)
    return expected, count


def test_sqlite_resolver_batch_matches_scalar(tmp_path):
    rng = np.random.default_rng(2026)
    pairs = rng.integers(1, 300, size=(2000, 2), dtype=np.int64)
    scalar = Equivalences(tmp_path / "scalar.sqlite")
    batched = Equivalences(tmp_path / "batch.sqlite")
    try:
        for a, b in pairs:
            scalar.union(int(a), int(b))
        for start in range(0, len(pairs), 137):
            batched.union_many(pairs[start : start + 137])
        scalar.flush()
        batched.flush()
        keys = np.arange(1, 300, dtype=np.int64)
        assert scalar.merges == batched.merges
        scalar_map = {int(k): scalar.find(int(k))[2] for k in keys}
        batch_map = {int(k): batched.find(int(k))[2] for k in keys}
        assert scalar_map == batch_map
    finally:
        scalar.close()
        batched.close()


def test_many_independent_components_cross_chunk_boundaries():
    image = np.zeros((48, 40, 40), dtype=bool)
    image[:, ::3, ::3] = True
    expected_components = len(range(0, 40, 3)) ** 2
    out = np.empty(image.shape, dtype=np.int64)
    result = label(image, out, chunks=(6, 10, 10), connectivity=1)
    expected, count = _reference(image)
    np.testing.assert_array_equal(out, expected)
    assert count == expected_components
    assert result.num_components == expected_components
