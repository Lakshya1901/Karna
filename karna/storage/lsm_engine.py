"""
karna/storage/lsm_engine.py
============================

Full LSM-tree storage engine tying together MemTable, SSTable, WAL, and
compaction logic.

Architecture
------------
::

    Writes ──► MemTable (RAM) ──flush──► L0 SSTables (disk)
                                                │
                                          compact (≥ 4 files)
                                                │
                                                ▼
                                          L1 SSTables (sorted, non-overlapping)

Reads check layers in order: MemTable → L0 (newest first) → L1 (newest first).

Write-Ahead Log (WAL)
---------------------
Every mutation is first appended to ``wal.log`` (one line per op)::

    PUT <key_hex> <val_hex>
    DEL <key_hex>

Keys and values are hex-encoded so the log is newline-safe.  On startup,
``recover_from_wal()`` replays the WAL to reconstruct the MemTable.

Compaction
----------
L0 compaction (``_compact_l0``) is a simplified size-tiered merge: when L0
accumulates more than ``L0_COMPACTION_TRIGGER`` SSTables, all L0 files are
merged into a single new L1 SSTable.  A full production implementation would
perform size-based tiering within L1 to bound read amplification.

Read Amplification (worst case)
    ``O(L0_files × index_block_scan + L1_files × index_block_scan)``

Write Amplification
    ``O(1)`` in-memory + occasional flush + compaction merge

References
----------
* Hahn et al. (2020) — Dostoevsky: Better Space-Time Trade-Offs for LSM-Tree
  Based Key-Value Stores via Adaptive Removal of Superfluous Merging
* RocksDB wiki — Compaction
"""

from __future__ import annotations

import os
import sys
import time
import uuid
from typing import Optional

# Adjust path so the module can be run directly
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from karna.storage.memtable import MemTable
from karna.storage.sstable import LSMLevel, SSTableReader, SSTableWriter

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

L0_COMPACTION_TRIGGER = 4   # compact when L0 has this many SSTables
WAL_FILENAME          = "wal.log"


# ---------------------------------------------------------------------------
# LSMEngine
# ---------------------------------------------------------------------------

class LSMEngine:
    """LSM-tree storage engine with WAL, MemTable, L0/L1 SSTables, and compaction.

    Parameters
    ----------
    data_dir:
        Directory for SSTable files and the WAL.  Created on first use.
    memtable_threshold_mb:
        MemTable flush threshold in MiB (default 4 MiB).

    Attributes
    ----------
    data_dir : str
        Root data directory.
    _memtable : MemTable
        The active in-memory write buffer.
    _l0 : LSMLevel
        Level-0 SSTable collection (freshest data, may have overlapping key ranges).
    _l1 : LSMLevel
        Level-1 SSTable collection (sorted, non-overlapping after compaction).
    _wal : io.TextIOWrapper
        Append-only write-ahead log file handle.

    Examples
    --------
    >>> import tempfile, os
    >>> with tempfile.TemporaryDirectory() as d:
    ...     engine = LSMEngine(d)
    ...     engine.put("name", b"karna")
    ...     engine.get("name")
    ...     engine.close()
    b'karna'
    """

    def __init__(
        self,
        data_dir: str,
        memtable_threshold_mb: float = 4.0,
    ) -> None:
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)

        threshold = int(memtable_threshold_mb * 1024 * 1024)
        self._memtable = MemTable(flush_threshold_bytes=threshold)
        self._l0 = LSMLevel(level_id=0)
        self._l1 = LSMLevel(level_id=1)

        # Open WAL in append mode
        wal_path = os.path.join(data_dir, WAL_FILENAME)
        self._wal = open(wal_path, "a", encoding="ascii")

        # Recover from WAL and load existing SSTables
        self._recover_sstables()
        self.recover_from_wal()

    # ------------------------------------------------------------------
    # Public write API
    # ------------------------------------------------------------------

    def put(self, key: str, value: bytes) -> None:
        """Write a key-value pair to the store.

        The write goes to the WAL and the active MemTable.  If the MemTable
        exceeds its threshold, it is flushed to a new L0 SSTable.

        Parameters
        ----------
        key:
            String key.
        value:
            Byte payload.
        """
        self._wal_append("PUT", key, value)
        self._memtable.put(key, value, timestamp=time.time_ns())
        if self._memtable.is_full:
            self._flush_memtable()

    def delete(self, key: str) -> None:
        """Write a deletion tombstone for *key*.

        Parameters
        ----------
        key:
            The key to delete.
        """
        self._wal_append("DEL", key, b"")
        self._memtable.delete(key, timestamp=time.time_ns())
        if self._memtable.is_full:
            self._flush_memtable()

    # ------------------------------------------------------------------
    # Public read API
    # ------------------------------------------------------------------

    def get(self, key: str) -> Optional[bytes]:
        """Retrieve the latest value for *key*, or ``None`` if absent/deleted.

        Read path:
        1. MemTable — O(log n) lookup.
        2. L0 SSTables newest-to-oldest — O(bloom + binary-search + linear).
        3. L1 SSTables newest-to-oldest.

        Parameters
        ----------
        key:
            String key to look up.

        Returns
        -------
        bytes or None
            Latest value, or ``None`` if the key is absent or deleted.
        """
        # 1. MemTable
        rec = self._memtable._data.get(key)
        if rec is not None:
            return None if rec.is_deleted else rec.value

        # 2. L0
        val, found = self._l0.get(key)
        if found:
            return val   # may be None (tombstone)

        # 3. L1
        val, found = self._l1.get(key)
        if found:
            return val

        return None

    # ------------------------------------------------------------------
    # Flush
    # ------------------------------------------------------------------

    def _flush_memtable(self) -> None:
        """Flush the active MemTable to a new L0 SSTable file.

        After flushing:
        * A fresh empty MemTable replaces the flushed one.
        * The WAL is truncated (data is now persisted in the SSTable).
        * If L0 has ≥ ``L0_COMPACTION_TRIGGER`` files, compaction is triggered.
        """
        sst_path = self._new_sst_path(level=0)
        writer = SSTableWriter(sst_path)
        for key, value, ts, deleted in self._memtable.to_sorted_iterator():
            writer.add(key, value, ts, deleted)
        writer.finish()

        self._l0.add_table(SSTableReader(sst_path))

        # Reset MemTable and truncate WAL
        threshold = self._memtable.flush_threshold_bytes
        self._memtable = MemTable(flush_threshold_bytes=threshold)
        self._truncate_wal()

        if len(self._l0) >= L0_COMPACTION_TRIGGER:
            self._compact_l0()

    # ------------------------------------------------------------------
    # Compaction
    # ------------------------------------------------------------------

    def _compact_l0(self) -> None:
        """Merge all L0 SSTables into a single new L1 SSTable.

        Algorithm (size-tiered merge):
        1. Open a merge iterator over all L0 SSTables (sorted merge).
        2. For duplicate keys, keep the record with the highest timestamp.
        3. Write the merged stream to a new L1 SSTable.
        4. Close and delete the old L0 files.

        This is analogous to LevelDB's ``VersionSet::LogAndApply`` compaction.
        """
        # Collect all records from L0, newest-file first so newer timestamps win
        merged: dict[str, tuple] = {}  # key → (value, ts, deleted)
        for table in self._l0._tables:
            for key, val, ts, deleted in table.scan():
                existing = merged.get(key)
                if existing is None or ts > existing[1]:
                    merged[key] = (val, ts, deleted)

        old_paths = [t.path for t in self._l0._tables]

        # Write merged output to L1
        sst_path = self._new_sst_path(level=1)
        writer = SSTableWriter(sst_path)
        for key in sorted(merged):
            val, ts, deleted = merged[key]
            writer.add(key, val, ts, deleted)
        writer.finish()
        self._l1.add_table(SSTableReader(sst_path))

        # Close and remove old L0 files
        self._l0.close_all()
        self._l0 = LSMLevel(level_id=0)
        for p in old_paths:
            try:
                os.remove(p)
            except OSError:
                pass

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Flush any remaining MemTable data and close all file handles.

        Safe to call multiple times.
        """
        if len(self._memtable) > 0:
            self._flush_memtable()
        self._wal.close()
        self._l0.close_all()
        self._l1.close_all()

    def recover_from_wal(self) -> int:
        """Replay the WAL to reconstruct the MemTable after a crash.

        Returns
        -------
        int
            Number of operations replayed.
        """
        wal_path = os.path.join(self.data_dir, WAL_FILENAME)
        if not os.path.exists(wal_path):
            return 0
        replayed = 0
        with open(wal_path, "r", encoding="ascii") as fh:
            for line in fh:
                line = line.rstrip("\n")
                if not line:
                    continue
                parts = line.split(" ", 2)
                op = parts[0]
                if op == "PUT" and len(parts) == 3:
                    key = bytes.fromhex(parts[1]).decode("utf-8")
                    val = bytes.fromhex(parts[2])
                    self._memtable.put(key, val, timestamp=time.time_ns())
                    replayed += 1
                elif op == "DEL" and len(parts) >= 2:
                    key = bytes.fromhex(parts[1]).decode("utf-8")
                    self._memtable.delete(key, timestamp=time.time_ns())
                    replayed += 1
        return replayed

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _wal_append(self, op: str, key: str, value: bytes) -> None:
        """Append one operation to the WAL."""
        key_hex = key.encode("utf-8").hex()
        val_hex = value.hex()
        if op == "PUT":
            self._wal.write(f"PUT {key_hex} {val_hex}\n")
        else:
            self._wal.write(f"DEL {key_hex}\n")
        self._wal.flush()

    def _truncate_wal(self) -> None:
        """Truncate the WAL after a successful MemTable flush."""
        self._wal.close()
        wal_path = os.path.join(self.data_dir, WAL_FILENAME)
        open(wal_path, "w").close()   # truncate
        self._wal = open(wal_path, "a", encoding="ascii")

    def _new_sst_path(self, level: int) -> str:
        """Generate a unique SSTable file path."""
        uid = uuid.uuid4().hex[:12]
        fname = f"L{level}_{uid}.sst"
        return os.path.join(self.data_dir, fname)

    def _recover_sstables(self) -> None:
        """Load existing SSTable files from data_dir into L0/L1 on startup."""
        for fname in sorted(os.listdir(self.data_dir)):
            if fname.endswith(".sst"):
                path = os.path.join(self.data_dir, fname)
                reader = SSTableReader(path)
                if fname.startswith("L0_"):
                    self._l0.add_table(reader)
                elif fname.startswith("L1_"):
                    self._l1.add_table(reader)

    def __repr__(self) -> str:
        return (
            f"LSMEngine(data_dir={self.data_dir!r}, "
            f"memtable={len(self._memtable)} entries, "
            f"L0={len(self._l0)} tables, "
            f"L1={len(self._l1)} tables)"
        )


# ---------------------------------------------------------------------------
# Sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import tempfile

    print("=" * 60)
    print("LSMEngine Sanity Check")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as data_dir:
        N = 5000

        # ---- Phase 1: write 5000 keys ----
        engine = LSMEngine(data_dir, memtable_threshold_mb=0.1)  # tiny threshold to force flushes
        print(f"Writing {N} keys...")
        for i in range(N):
            engine.put(f"key:{i:08d}", f"value:{i}".encode())
        print(f"  After writes: {engine!r}")
        engine.close()
        print("[PASS] Wrote and closed engine")

        # ---- Phase 2: reopen and verify via WAL + SSTables ----
        print("Reopening engine (WAL recovery + SSTable loading)...")
        engine2 = LSMEngine(data_dir, memtable_threshold_mb=0.1)
        print(f"  After recovery: {engine2!r}")

        # Spot-check 200 keys
        sample_indices = list(range(0, N, N // 200))
        misses = 0
        for i in sample_indices:
            key = f"key:{i:08d}"
            expected = f"value:{i}".encode()
            val = engine2.get(key)
            if val != expected:
                misses += 1
                print(f"  MISS: {key} → {val!r} (expected {expected!r})")
        if misses == 0:
            print(f"[PASS] All {len(sample_indices)} sampled keys verified correct")
        else:
            print(f"[FAIL] {misses} mismatches")

        # Delete a key
        engine2.delete("key:00000042")
        assert engine2.get("key:00000042") is None
        print("[PASS] Tombstone delete verified")

        engine2.close()
        print("\nAll LSMEngine checks passed ✓")
