"""Disk-backed disjoint sets. Absent rows are implicit singleton components."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np

_SQL_BATCH = 400


class Equivalences:
    """Union by size, path compression, and stable minimum-key representatives.

    Only components that participate in an equivalence need database rows.
    All other provisional labels remain implicit singleton sets.  The SQLite
    page cache is bounded; no global Python dictionary of components is built.

    Boundary equivalences are resolved in bounded batches.  A batch is scoped
    to one boundary comparison, so memory scales with boundary area rather
    than with the total number of components in the image.
    """

    def __init__(self, path: Path, cache_bytes: int = 8 * 1024 * 1024):
        self.db = sqlite3.connect(path)
        self.db.execute(f"PRAGMA cache_size = -{max(1, cache_bytes // 1024)}")
        self.db.execute("PRAGMA temp_store = FILE")
        self.db.execute(
            "CREATE TABLE nodes (id INTEGER PRIMARY KEY, parent INTEGER NOT NULL, "
            "size INTEGER NOT NULL, minimum INTEGER NOT NULL)"
        )
        self.merges = 0
        self._pending = 0

    def close(self) -> None:
        try:
            self.db.commit()
        finally:
            self.db.close()

    def _row(self, key: int) -> tuple[int, int, int]:
        row = self.db.execute(
            "SELECT parent, size, minimum FROM nodes WHERE id = ?", (key,)
        ).fetchone()
        return (key, 1, key) if row is None else row

    def find(self, key: int) -> tuple[int, int, int]:
        """Return (root, set size, smallest original key)."""
        path: list[int] = []
        current = int(key)
        while True:
            parent, size, minimum = self._row(current)
            if parent == current:
                break
            path.append(current)
            current = parent
        for node in path:
            self.db.execute("UPDATE nodes SET parent = ? WHERE id = ?", (current, node))
        return current, size, minimum

    def _load_rows(self, keys: set[int]) -> dict[int, list[int]]:
        rows: dict[int, list[int]] = {}
        pending = set(keys)
        while pending:
            batch = list(pending)[:_SQL_BATCH]
            pending.difference_update(batch)
            marks = ",".join("?" for _ in batch)
            fetched = self.db.execute(
                f"SELECT id, parent, size, minimum FROM nodes WHERE id IN ({marks})", batch
            ).fetchall()
            found = {int(row[0]) for row in fetched}
            for key, parent, size, minimum in fetched:
                rows[int(key)] = [int(parent), int(size), int(minimum)]
            for key in batch:
                if key not in found:
                    rows[key] = [key, 1, key]
            for key, (parent, _, _) in tuple(rows.items()):
                if parent not in rows:
                    pending.add(parent)
        return rows

    def union_many(self, pairs: np.ndarray) -> int:
        """Resolve a bounded array of ``(a, b)`` equivalences efficiently.

        Returns the number of newly merged disjoint sets. Reads and writes are
        batched so high-boundary-component workloads do not issue several SQL
        statements per pair.
        """
        pairs = np.asarray(pairs, dtype=np.int64)
        if pairs.size == 0:
            return 0
        pairs = pairs.reshape(-1, 2)
        keys = {int(x) for x in pairs.ravel()}
        rows = self._load_rows(keys)
        dirty: set[int] = set()

        def find_local(key: int) -> tuple[int, int, int]:
            path: list[int] = []
            current = key
            while True:
                parent, size, minimum = rows[current]
                if parent == current:
                    break
                path.append(current)
                current = parent
            for node in path:
                if rows[node][0] != current:
                    rows[node][0] = current
                    dirty.add(node)
            return current, size, minimum

        merged = 0
        for a, b in pairs:
            ra, sa, ma = find_local(int(a))
            rb, sb, mb = find_local(int(b))
            if ra == rb:
                continue
            if sa < sb or (sa == sb and ra > rb):
                ra, rb = rb, ra
                sa, sb = sb, sa
                ma, mb = mb, ma
            rows[ra] = [ra, sa + sb, min(ma, mb)]
            rows[rb] = [ra, 0, mb]
            dirty.add(ra)
            dirty.add(rb)
            merged += 1

        if dirty:
            self.db.executemany(
                "INSERT INTO nodes VALUES (?, ?, ?, ?) "
                "ON CONFLICT(id) DO UPDATE SET parent=excluded.parent, "
                "size=excluded.size, minimum=excluded.minimum",
                ((key, *rows[key]) for key in dirty),
            )
        self.merges += merged
        self._pending += merged
        if self._pending >= 4096:
            self.flush()
        return merged

    def union(self, a: int, b: int) -> bool:
        return bool(self.union_many(np.asarray([[a, b]], dtype=np.int64)))

    def flush(self) -> None:
        self.db.commit()
        self._pending = 0

    def replacements(self, keys: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Find changed canonical IDs among one chunk's sorted, unique keys."""
        old: list[int] = []
        new: list[int] = []
        for start in range(0, len(keys), _SQL_BATCH):
            batch = [int(x) for x in keys[start : start + _SQL_BATCH] if x != 0]
            if not batch:
                continue
            marks = ",".join("?" for _ in batch)
            rows = self.db.execute(
                f"SELECT id FROM nodes WHERE id IN ({marks})", batch
            ).fetchall()
            for (key,) in rows:
                _, _, minimum = self.find(key)
                if minimum != key:
                    old.append(key)
                    new.append(minimum)
        if not old:
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
        order = np.argsort(old)
        return np.asarray(old, dtype=np.int64)[order], np.asarray(new, dtype=np.int64)[order]
