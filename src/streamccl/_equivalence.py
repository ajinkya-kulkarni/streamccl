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
        # The resolver database is private, ephemeral scratch state.  It never
        # needs crash recovery and is not shared across processes, so avoid
        # journal/fsync/lock-manager overhead while keeping the page cache bounded.
        self.db.execute("PRAGMA locking_mode = EXCLUSIVE")
        self.db.execute("PRAGMA journal_mode = OFF")
        self.db.execute("PRAGMA synchronous = OFF")
        self.db.execute(f"PRAGMA cache_size = -{max(1, cache_bytes // 1024)}")
        self.db.execute("PRAGMA temp_store = FILE")
        self.db.execute(
            "CREATE TABLE nodes (id INTEGER PRIMARY KEY, parent INTEGER NOT NULL, "
            "size INTEGER NOT NULL, minimum INTEGER NOT NULL)"
        )
        self.merges = 0
        self._pending = 0
        self._finalized = False

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
        # Parents may live outside the boundary batch (for example a filament
        # already merged across previous chunks).  Fetch missing ancestor rows
        # in bounded SQL batches until every cached path reaches a known root
        # or an implicit singleton.
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

        Returns the number of newly merged disjoint sets.  Reads and writes are
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
        if merged:
            self._finalized = False
        if self._pending >= 4096:
            self.flush()
        return merged

    def union(self, a: int, b: int) -> bool:
        return bool(self.union_many(np.asarray([[a, b]], dtype=np.int64)))

    def flush(self) -> None:
        self.db.commit()
        self._pending = 0

    def finalize(self) -> None:
        """Flatten explicit parent chains using bounded batched pointer jumping.

        The resolver deliberately keeps its global state on disk.  A single
        correlated SQL ``UPDATE`` over a million nodes is pathologically slow
        in SQLite, so finalization scans only nodes whose parent is not yet a
        root and updates them in bounded batches.  Union-by-size keeps the
        forest shallow, making this a small number of sequential passes while
        memory remains independent of the total node count.
        """
        if self._finalized:
            return
        self.flush()
        batch_size = 16_384
        while True:
            changed = 0
            last_id = -1
            while True:
                rows = self.db.execute(
                    "SELECT c.id, p.parent "
                    "FROM nodes AS c JOIN nodes AS p ON p.id = c.parent "
                    "WHERE c.id > ? AND c.parent != c.id AND p.parent != p.id "
                    "ORDER BY c.id LIMIT ?",
                    (last_id, batch_size),
                ).fetchall()
                if not rows:
                    break
                self.db.executemany(
                    "UPDATE nodes SET parent = ? WHERE id = ?",
                    ((int(parent), int(key)) for key, parent in rows),
                )
                changed += len(rows)
                last_id = int(rows[-1][0])
            self.db.commit()
            if changed == 0:
                break
        self._finalized = True

    def materialize_replacements(
        self, max_bytes: int
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """Return the complete canonical map when it fits in ``max_bytes``.

        Two int64 arrays cost exactly 16 bytes per changed provisional label.
        If the map would exceed the caller's spare-memory budget, return
        ``None`` and let the caller use chunk-local disk-backed lookups.
        """
        self.finalize()
        row = self.db.execute(
            "SELECT count(*) FROM nodes AS c "
            "JOIN nodes AS r ON r.id = c.parent "
            "WHERE r.minimum != c.id"
        ).fetchone()
        count = int(row[0])
        if count == 0:
            empty = np.empty(0, dtype=np.int64)
            return empty, empty.copy()
        if count * 16 > max_bytes:
            return None
        old = np.empty(count, dtype=np.int64)
        new = np.empty(count, dtype=np.int64)
        cursor = self.db.execute(
            "SELECT c.id, r.minimum FROM nodes AS c "
            "JOIN nodes AS r ON r.id = c.parent "
            "WHERE r.minimum != c.id ORDER BY c.id"
        )
        offset = 0
        while True:
            rows = cursor.fetchmany(16_384)
            if not rows:
                break
            n = len(rows)
            old[offset : offset + n] = [int(key) for key, _ in rows]
            new[offset : offset + n] = [int(minimum) for _, minimum in rows]
            offset += n
        if offset != count:
            raise RuntimeError("resolver replacement count changed during materialization")
        return old, new

    def replacements(self, keys: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Find changed canonical IDs among one chunk's sorted, unique keys."""
        self.finalize()
        old: list[int] = []
        new: list[int] = []
        # Finalization makes canonical IDs a direct column lookup, avoiding a
        # parent-chain query for every provisional component in every chunk.
        for start in range(0, len(keys), _SQL_BATCH):
            batch = [int(x) for x in keys[start : start + _SQL_BATCH] if x != 0]
            if not batch:
                continue
            marks = ",".join("?" for _ in batch)
            rows = self.db.execute(
                f"SELECT c.id, r.minimum FROM nodes AS c "
                f"JOIN nodes AS r ON r.id = c.parent "
                f"WHERE c.id IN ({marks}) AND r.minimum != c.id",
                batch,
            ).fetchall()
            for key, minimum in rows:
                old.append(int(key))
                new.append(int(minimum))
        if not old:
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
        order = np.argsort(old)
        return np.asarray(old, dtype=np.int64)[order], np.asarray(new, dtype=np.int64)[order]
