"""Disk-backed disjoint sets. Absent rows are implicit singleton components."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np


class Equivalences:
    """Union by size, path compression, and stable minimum-key representatives.

    Only components that participate in an equivalence need database rows.
    All other provisional labels remain implicit singleton sets. The SQLite
    page cache is bounded; no global Python dictionary of components is built.
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

    def union(self, a: int, b: int) -> bool:
        ra, sa, ma = self.find(int(a))
        rb, sb, mb = self.find(int(b))
        if ra == rb:
            return False
        if sa < sb or (sa == sb and ra > rb):
            ra, rb = rb, ra
            sa, sb = sb, sa
        self.db.execute(
            "INSERT INTO nodes VALUES (?, ?, ?, ?) "
            "ON CONFLICT(id) DO UPDATE SET parent=excluded.parent, "
            "size=excluded.size, minimum=excluded.minimum",
            (ra, ra, sa + sb, min(ma, mb)),
        )
        self.db.execute(
            "INSERT INTO nodes VALUES (?, ?, 0, ?) "
            "ON CONFLICT(id) DO UPDATE SET parent=excluded.parent, size=0",
            (rb, ra, mb),
        )
        self.merges += 1
        self._pending += 1
        if self._pending >= 4096:
            self.flush()
        return True

    def flush(self) -> None:
        self.db.commit()
        self._pending = 0

    def replacements(self, keys: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Find changed canonical IDs among one chunk's sorted, unique keys."""
        old: list[int] = []
        new: list[int] = []
        for start in range(0, len(keys), 400):
            batch = [int(x) for x in keys[start : start + 400] if x != 0]
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
