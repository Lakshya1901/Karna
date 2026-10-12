"""
karna/storage/memtable.py
=========================

In-memory sorted table (MemTable) — the write buffer at the top of the
LSM-tree hierarchy.

Design Notes
------------
A production LSM-tree (LevelDB, RocksDB) uses a **skip list** as the MemTable
because it provides O(log n) amortised reads/writes with cache-friendly
sequential iteration.  A skip list is a probabilistic, layered linked-list
structure that achieves the same asymptotic complexity as a balanced BST
without requiring rotations.

This prototype replaces the skip list with :class:`sortedcontainers.SortedDict`
(a pure-Python B-tree variant) when available, falling back to a plain
``dict`` with a sort on iteration.  Both expose the same public API so the
rest of the engine is insulated from the choice.

Lifecycle
---------
1. Writes go to the active MemTable and the WAL simultaneously.
2. When ``is_full`` is ``True`` the engine swaps in a fresh MemTable and
   schedules the old one for flushing to an L0 SSTable.
3. On flush, ``to_sorted_iterator()`` drives the SSTableWriter in O(n) time.

References
----------
* O'Neil et al. (1996) — The Log-Structured Merge-Tree (LSM-Tree)
* Pugh (1990) — Skip Lists: A Probabilistic Alternative to Balanced Trees
* SortedContainers docs — https://grantjenks.com/docs/sortedcontainers/
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from typing import Iterator, Optional, Tuple

# ---------------------------------------------------------------------------
# Optional fast sorted container
# ---------------------------------------------------------------------------
try:
    from sortedcontainers import SortedDict  # type: ignore

    _HAS_SORTED_CONTAINERS = True
except ImportError:  # pragma: no cover
    _HAS_SORTED_CONTAINERS = False


# ---------------------------------------------------------------------------
# Internal record
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class _Record:
    """A single versioned record stored in the MemTable.

    Parameters
    ----------
    value:
        Raw byte payload; ``None`` when the record is a deletion tombstone.
    timestamp:
        Monotonic integer timestamp (nanoseconds or logical clock).  Later
        writes with the same key must have a strictly greater timestamp so
        that the most-recent record wins during compaction.
    is_deleted:
        ``True`` if this record is a tombstone (written by :meth:`delete`).
    """

    value: Optional[bytes]
    timestamp: int
    is_deleted: bool = False


# ---------------------------------------------------------------------------
# MemTable
# ---------------------------------------------------------------------------

class MemTable:
    """Write-buffer at the top of the LSM-tree hierarchy.

    Keys are stored in sorted order to enable O(n) merge during SSTable flush.
    Each key maps to the *latest* :class:`_Record`; older versions are
    discarded immediately (single-version in-memory design matching LevelDB).

    Parameters
    ----------
    flush_threshold_bytes:
        Maximum estimated byte size before ``is_full`` returns ``True``.
        Defaults to 4 MiB, matching RocksDB's ``write_buffer_size`` default.

    Attributes
    ----------
    flush_threshold_bytes : int
        Configurable flush threshold (bytes).
    _data : SortedDict | dict
        Underlying sorted (or plain) dictionary of key → :class:`_Record`.
    _size_bytes : int
        Running byte-size estimate updated on every write.
    _write_count : int
        Total number of write operations (puts + deletes) since creation.

    Examples
    --------
    >>> mt = MemTable()
    >>> mt.put("hello", b"world", timestamp=1)
    >>> mt.get("hello")
    b'world'
    >>> mt.delete("hello", timestamp=2)
    >>> mt.get("hello") is None
    True
    """

    flush_threshold_bytes: int = 4 * 1024 * 1024  # 4 MiB

    def __init__(self, flush_threshold_bytes: int = 4 * 1024 * 1024) -> None:
        self.flush_threshold_bytes = flush_threshold_bytes
        self._data: dict = SortedDict() if _HAS_SORTED_CONTAINERS else {}
        self._size_bytes: int = 0
        self._write_count: int = 0

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def put(self, key: str, value: bytes, timestamp: Optional[int] = None) -> None:
        """Insert or update a key-value pair.

        Parameters
        ----------
        key:
            UTF-8 string key.
        value:
            Arbitrary byte payload.
        timestamp:
            Logical clock value.  If ``None``, uses ``time.time_ns()``.
        """
        if timestamp is None:
            timestamp = time.time_ns()
        old = self._data.get(key)
        if old is not None:
            # Reclaim the space occupied by the previous record
            self._size_bytes -= _record_byte_size(key, old)
        record = _Record(value=value, timestamp=timestamp, is_deleted=False)
        self._data[key] = record
        self._size_bytes += _record_byte_size(key, record)
        self._write_count += 1

    def get(self, key: str) -> Optional[bytes]:
        """Retrieve the value for *key*, or ``None`` if absent/deleted.

        Parameters
        ----------
        key:
            The key to look up.

        Returns
        -------
        bytes or None
            The stored value, or ``None`` if the key does not exist or has
            been deleted (tombstone present).
        """
        record = self._data.get(key)
        if record is None or record.is_deleted:
            return None
        return record.value

    def delete(self, key: str, timestamp: Optional[int] = None) -> None:
        """Write a deletion tombstone for *key*.

        The key is not removed from ``_data``; instead a tombstone record is
        stored so that compaction can suppress older versions in lower levels.

        Parameters
        ----------
        key:
            The key to tombstone.
        timestamp:
            Logical clock value.  Defaults to ``time.time_ns()``.
        """
        if timestamp is None:
            timestamp = time.time_ns()
        old = self._data.get(key)
        if old is not None:
            self._size_bytes -= _record_byte_size(key, old)
        record = _Record(value=None, timestamp=timestamp, is_deleted=True)
        self._data[key] = record
        self._size_bytes += _record_byte_size(key, record)
        self._write_count += 1

    def contains(self, key: str) -> bool:
        """Return ``True`` if the key is present (not tombstoned)."""
        record = self._data.get(key)
        return record is not None and not record.is_deleted

    # ------------------------------------------------------------------
    # Size / capacity
    # ------------------------------------------------------------------

    @property
    def size_bytes(self) -> int:
        """Estimated memory footprint in bytes (updated incrementally)."""
        return self._size_bytes

    @property
    def is_full(self) -> bool:
        """``True`` when the MemTable should be flushed to disk."""
        return self._size_bytes >= self.flush_threshold_bytes

    def __len__(self) -> int:
        """Number of unique keys (including tombstones)."""
        return len(self._data)

    # ------------------------------------------------------------------
    # Iteration
    # ------------------------------------------------------------------

    def to_sorted_iterator(
        self,
    ) -> Iterator[Tuple[str, Optional[bytes], int, bool]]:
        """Yield all records in ascending key order.

        Yields
        ------
        tuple of (key, value, timestamp, is_deleted)
            ``value`` is ``None`` for tombstone records.
        """
        keys = self._data.keys() if _HAS_SORTED_CONTAINERS else sorted(self._data.keys())
        for key in keys:
            rec: _Record = self._data[key]
            yield key, rec.value, rec.timestamp, rec.is_deleted

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        backend = "SortedDict" if _HAS_SORTED_CONTAINERS else "dict"
        return (
            f"MemTable(entries={len(self)}, "
            f"size_bytes={self._size_bytes}, "
            f"backend={backend})"
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _record_byte_size(key: str, record: _Record) -> int:
    """Rough byte-size estimate for a single key-record pair.

    This mirrors the approach used by RocksDB's ``KeyValueSize`` helper:
    key UTF-8 bytes + value bytes + fixed overhead for metadata fields.

    Parameters
    ----------
    key:
        The record key.
    record:
        The associated :class:`_Record`.

    Returns
    -------
    int
        Estimated bytes consumed by this entry.
    """
    key_bytes = len(key.encode("utf-8"))
    val_bytes = len(record.value) if record.value is not None else 0
    overhead = 24  # timestamp (8B) + is_deleted (1B) + Python object overhead (~15B)
    return key_bytes + val_bytes + overhead


# ---------------------------------------------------------------------------
# Sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("MemTable Sanity Check")
    print(f"  sortedcontainers available: {_HAS_SORTED_CONTAINERS}")
    print("=" * 60)

    mt = MemTable(flush_threshold_bytes=1024)  # tiny threshold for testing

    # Basic put / get
    mt.put("apple", b"red", timestamp=1)
    mt.put("banana", b"yellow", timestamp=2)
    mt.put("cherry", b"dark_red", timestamp=3)
    assert mt.get("apple") == b"red", "put/get failed"
    assert mt.get("banana") == b"yellow", "put/get failed"
    print("[PASS] Basic put / get")

    # Update
    mt.put("apple", b"green", timestamp=4)
    assert mt.get("apple") == b"green", "update failed"
    print("[PASS] Update (overwrite)")

    # Delete (tombstone)
    mt.delete("banana", timestamp=5)
    assert mt.get("banana") is None, "delete (tombstone) failed"
    print("[PASS] Tombstone delete")

    # Sorted iteration order
    keys_in_order = [k for k, _, _, _ in mt.to_sorted_iterator()]
    assert keys_in_order == sorted(keys_in_order), "iteration not sorted"
    print(f"[PASS] Sorted iteration: {keys_in_order}")

    # Tombstone visible in iteration
    tombstones = [(k, d) for k, _, _, d in mt.to_sorted_iterator()]
    banana_entry = next(e for e in tombstones if e[0] == "banana")
    assert banana_entry[1] is True, "tombstone not in iterator"
    print("[PASS] Tombstone present in iterator")

    # Size tracking
    print(f"  size_bytes = {mt.size_bytes}")
    assert mt.size_bytes > 0
    print("[PASS] size_bytes > 0")

    # is_full with very small threshold
    tiny = MemTable(flush_threshold_bytes=10)
    tiny.put("x" * 100, b"y" * 100, timestamp=1)
    assert tiny.is_full, "is_full should be True"
    print("[PASS] is_full triggered")

    # Bulk load
    big = MemTable()
    for i in range(10_000):
        big.put(f"key:{i:08d}", f"value:{i}".encode(), timestamp=i)
    assert len(big) == 10_000
    # Verify first key in sorted order
    first_key = next(k for k, _, _, _ in big.to_sorted_iterator())
    assert first_key == "key:00000000", f"unexpected first key: {first_key}"
    print(f"[PASS] Bulk load of 10,000 entries; first key = {first_key!r}")

    print(f"\n{big!r}")
    print("\nAll MemTable checks passed ✓")
