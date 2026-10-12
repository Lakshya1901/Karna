"""
karna/storage/sstable.py
========================

Sorted String Table (SSTable) — the on-disk immutable sorted file format at
the heart of every LSM-tree implementation.

File Format
-----------
Each SSTable file is composed of three sections written sequentially:

1. **Data Block** — Variable-length records, one per key-value pair::

       [key_len : 4 B LE uint32]
       [key     : key_len bytes (UTF-8)]
       [val_len : 4 B LE uint32]
       [val     : val_len bytes  (0 bytes for tombstones)]
       [ts      : 8 B LE int64  (nanosecond timestamp)]
       [deleted : 1 B           (0x01 = tombstone)]

2. **Sparse Index** — Every INDEX_INTERVAL-th key is stored with its byte
   offset into the data block.  On a read, we binary-search the sparse index
   to find the nearest preceding entry, then scan linearly from that offset.
   This mirrors LevelDB's two-level index design.

3. **Footer** — Two 8-byte little-endian integers:
       [index_offset : 8 B] — byte offset where the sparse index starts
       [bloom_offset : 8 B] — byte offset where the bloom filter starts

   Followed by 8 magic bytes ``b'KARNATBL'`` so we can detect corruption.

Bloom Filter
------------
A :class:`BloomFilter` with *k* = 3 hash functions and *m* = 1024 bits is
serialised as 128 bytes (1024 // 8) prepended with a 4-byte ``k`` field.
False-positive rate at expected occupancy n = 100 keys:
    p ≈ (1 - e^{-kn/m})^k ≈ 0.8 %

References
----------
* Bloom (1970) — Space/Time Trade-offs in Hash Coding with Allowable Errors
* Chang et al. (2006) — Bigtable: A Distributed Storage System for Structured Data
* LevelDB source — table/table_builder.cc
"""

from __future__ import annotations

import bisect
import hashlib
import math
import io
import os
import struct
from typing import Iterator, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_RECORD_HEADER_FMT = "<II"   # key_len, val_len (two uint32)
_TS_DELETED_FMT    = "<qB"   # timestamp (int64), deleted (uint8)
_RECORD_HEADER_SIZE = struct.calcsize(_RECORD_HEADER_FMT)  # 8 bytes
_TS_DELETED_SIZE    = struct.calcsize(_TS_DELETED_FMT)      # 9 bytes

_INDEX_FMT          = "<QI"  # (offset uint64, key_len uint32)
_INDEX_ENTRY_SIZE   = struct.calcsize(_INDEX_FMT)           # 12 bytes

_FOOTER_FMT         = "<QQ"  # (index_offset, bloom_offset)
_FOOTER_SIZE        = struct.calcsize(_FOOTER_FMT)          # 16 bytes
_MAGIC              = b"KARNATBL"                           # 8 bytes

INDEX_INTERVAL      = 16     # every 16th key goes in the sparse index


# ---------------------------------------------------------------------------
# BloomFilter
# ---------------------------------------------------------------------------

class BloomFilter:
    """Simple k-hash Bloom filter backed by a Python ``bytearray``.

    Uses *k* = 3 independent hash functions derived from SHA-256 by taking
    successive 64-bit sub-digests modulo *m*.  This is equivalent to the
    double-hashing technique (Kirsch & Mitzenmacher, 2006) and avoids the
    cost of running *k* independent hash algorithms.

    Parameters
    ----------
    m:
        Number of bits in the filter.  Must be a multiple of 8.
    k:
        Number of hash functions.

    Attributes
    ----------
    m : int
        Bit-width.
    k : int
        Hash function count.
    _bits : bytearray
        Backing bit array stored as bytes.

    Examples
    --------
    >>> bf = BloomFilter(m=1024, k=3)
    >>> bf.add("hello")
    >>> bf.query("hello")
    True
    >>> bf.query("definitely_not_inserted")  # may rarely be True (FP)
    False
    """

    def __init__(self, m: int = 1024, k: int = 3) -> None:
        if m % 8 != 0:
            raise ValueError(f"m must be a multiple of 8, got {m}")
        self.m = m
        self.k = k
        self._bits = bytearray(m // 8)

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------

    def _hashes(self, key: str) -> List[int]:
        """Return *k* bit positions for *key*.

        Kirsch–Mitzenmacher double hashing: two 64-bit halves of one SHA-256
        digest give ``h1 + i·h2 (mod m)`` for i = 0..k-1, which behaves like
        k independent hashes for any k (slicing fixed digest offsets repeats
        positions once k exceeds the number of slices).
        """
        digest = hashlib.sha256(key.encode("utf-8")).digest()
        h1 = int.from_bytes(digest[:8], "little")
        h2 = int.from_bytes(digest[8:16], "little") | 1  # odd → never degenerate
        return [(h1 + i * h2) % self.m for i in range(self.k)]

    @classmethod
    def for_capacity(cls, n: int, fpr: float = 0.01) -> "BloomFilter":
        """Optimally sized filter for *n* keys: m = -n·ln(p)/(ln 2)², k = (m/n)·ln 2."""
        n = max(1, n)
        m = max(64, math.ceil(-n * math.log(fpr) / (math.log(2) ** 2)))
        m += (-m) % 8
        k = max(1, round(m / n * math.log(2)))
        return cls(m=m, k=k)

    def add(self, key: str) -> None:
        """Insert *key* into the filter."""
        for pos in self._hashes(key):
            self._bits[pos >> 3] |= (1 << (pos & 7))

    def query(self, key: str) -> bool:
        """Return ``True`` if *key* is *possibly* in the set.

        A ``False`` return is a definitive miss; ``True`` may be a false
        positive with probability ≈ (1 - e^{-kn/m})^k.
        """
        return all(
            (self._bits[pos >> 3] >> (pos & 7)) & 1
            for pos in self._hashes(key)
        )

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_bytes(self) -> bytes:
        """Serialise filter to bytes: ``[k: 4B LE uint32][bits: m/8 bytes]``."""
        return struct.pack("<I", self.k) + bytes(self._bits)

    @classmethod
    def from_bytes(cls, data: bytes) -> "BloomFilter":
        """Deserialise a :class:`BloomFilter` produced by :meth:`to_bytes`."""
        k = struct.unpack_from("<I", data, 0)[0]
        bits = bytearray(data[4:])
        m = len(bits) * 8
        bf = cls(m=m, k=k)
        bf._bits = bits
        return bf

    def __repr__(self) -> str:
        return f"BloomFilter(m={self.m}, k={self.k})"


# ---------------------------------------------------------------------------
# SSTableWriter
# ---------------------------------------------------------------------------

class SSTableWriter:
    """Writes a single SSTable file from an iterator of sorted records.

    Usage::

        writer = SSTableWriter("/path/to/file.sst")
        for key, value, ts, deleted in sorted_iter:
            writer.add(key, value, ts, deleted)
        writer.finish()

    Parameters
    ----------
    path:
        Destination file path (created/overwritten).
    bloom_fpr:
        Target false-positive rate.  The filter is sized from the table's
        actual key count when :meth:`finish` runs (a fixed-size filter
        saturates on large tables and stops filtering anything).
    """

    def __init__(self, path: str, bloom_fpr: float = 0.01) -> None:
        self.path = path
        self._fh = open(path, "wb")
        self._bloom_fpr = bloom_fpr
        self._keys: List[str] = []
        self._index: List[Tuple[str, int]] = []   # (key, offset) pairs
        self._entry_count = 0
        self._offset = 0   # current write cursor in data block

    def add(
        self,
        key: str,
        value: Optional[bytes],
        timestamp: int,
        is_deleted: bool = False,
    ) -> None:
        """Append one record.  Keys MUST be provided in ascending order.

        Parameters
        ----------
        key:
            Record key (UTF-8 string).
        value:
            Record payload; pass ``None`` or ``b""`` for tombstones.
        timestamp:
            Write timestamp (int64).
        is_deleted:
            ``True`` to write a tombstone.
        """
        # Sparse index: record offset of every INDEX_INTERVAL-th entry
        if self._entry_count % INDEX_INTERVAL == 0:
            self._index.append((key, self._offset))

        self._keys.append(key)

        key_bytes = key.encode("utf-8")
        val_bytes  = value if (value is not None and not is_deleted) else b""

        header = struct.pack(_RECORD_HEADER_FMT, len(key_bytes), len(val_bytes))
        tail   = struct.pack(_TS_DELETED_FMT, timestamp, int(is_deleted))

        buf = header + key_bytes + val_bytes + tail
        self._fh.write(buf)
        self._offset += len(buf)
        self._entry_count += 1

    def finish(self) -> None:
        """Flush the sparse index and bloom filter, write footer, close file."""
        # ---- Sparse index ----
        index_offset = self._offset
        for idx_key, idx_off in self._index:
            key_bytes = idx_key.encode("utf-8")
            entry = struct.pack(_INDEX_FMT, idx_off, len(key_bytes)) + key_bytes
            self._fh.write(entry)

        # ---- Bloom filter ----
        bloom_offset = index_offset + (self._fh.tell() - index_offset)
        bloom = BloomFilter.for_capacity(len(self._keys), self._bloom_fpr)
        for key in self._keys:
            bloom.add(key)
        bloom_bytes = bloom.to_bytes()
        self._fh.write(bloom_bytes)

        # ---- Footer ----
        footer = struct.pack(_FOOTER_FMT, index_offset, bloom_offset)
        self._fh.write(footer)
        self._fh.write(_MAGIC)

        self._fh.flush()
        self._fh.close()

    def __enter__(self) -> "SSTableWriter":
        return self

    def __exit__(self, *_: object) -> None:
        self.finish()


# ---------------------------------------------------------------------------
# SSTableReader
# ---------------------------------------------------------------------------

class SSTableReader:
    """Random-access reader for a single SSTable file.

    Read path:
    1. Bloom filter check — definitive misses skip disk I/O entirely.
    2. Binary search on sparse index to find the nearest block offset.
    3. Linear scan from that offset until the key is found or surpassed.

    Parameters
    ----------
    path:
        Path to the SSTable file to open.
    """

    def __init__(self, path: str) -> None:
        self.path = path
        self._fh = open(path, "rb")
        self._load_footer()
        self._load_bloom()
        self._load_index()

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------

    def _load_footer(self) -> None:
        """Read the footer (last 24 bytes) and validate the magic marker."""
        self._fh.seek(-(len(_MAGIC) + _FOOTER_SIZE), 2)
        raw = self._fh.read(_FOOTER_SIZE)
        magic = self._fh.read(len(_MAGIC))
        if magic != _MAGIC:
            raise ValueError(f"Not a valid Karna SSTable: {self.path}")
        self._index_offset, self._bloom_offset = struct.unpack(_FOOTER_FMT, raw)

    def _load_bloom(self) -> None:
        """Deserialise the bloom filter from the file."""
        self._fh.seek(self._bloom_offset)
        bloom_end = self._index_offset - self._bloom_offset  # bytes of bloom
        # Actually bloom comes AFTER index; recalculate
        # bloom_offset is after index in file layout:
        # [data][index][bloom][footer][magic]
        self._fh.seek(self._bloom_offset)
        footer_magic_size = _FOOTER_SIZE + len(_MAGIC)
        file_size = self._fh.seek(0, 2)
        bloom_size = file_size - footer_magic_size - self._bloom_offset
        self._fh.seek(self._bloom_offset)
        self._bloom = BloomFilter.from_bytes(self._fh.read(bloom_size))

    def _load_index(self) -> None:
        """Load the sparse index into memory as (key, offset) pairs."""
        self._fh.seek(self._index_offset)
        end = self._bloom_offset
        self._index_keys: List[str]  = []
        self._index_offsets: List[int] = []
        while self._fh.tell() < end:
            raw = self._fh.read(_INDEX_ENTRY_SIZE)
            if len(raw) < _INDEX_ENTRY_SIZE:
                break
            off, klen = struct.unpack(_INDEX_FMT, raw)
            key = self._fh.read(klen).decode("utf-8")
            self._index_keys.append(key)
            self._index_offsets.append(off)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(self, key: str) -> Tuple[Optional[bytes], bool]:
        """Look up *key* in this SSTable.

        Returns
        -------
        tuple of (value, found)
            *value* is ``None`` for tombstones.  *found* is ``False`` when
            the key is definitively absent (bloom miss or scan exhausted).
        """
        if not self._bloom.query(key):
            return None, False

        # Binary-search sparse index for the largest key ≤ target key
        idx = bisect.bisect_right(self._index_keys, key) - 1
        scan_offset = self._index_offsets[idx] if idx >= 0 else 0

        self._fh.seek(scan_offset)
        while self._fh.tell() < self._index_offset:
            result = self._read_record()
            if result is None:
                break
            rkey, rval, _rts, rdeleted = result
            if rkey == key:
                if rdeleted:
                    return None, True   # tombstone — key was deleted
                return rval, True
            if rkey > key:
                break   # passed the key — not present
        return None, False

    def scan(
        self,
    ) -> Iterator[Tuple[str, Optional[bytes], int, bool]]:
        """Iterate over all records in key order (full table scan).

        Yields
        ------
        tuple of (key, value, timestamp, is_deleted)
        """
        self._fh.seek(0)
        while self._fh.tell() < self._index_offset:
            rec = self._read_record()
            if rec is None:
                break
            yield rec

    # ------------------------------------------------------------------
    # Low-level record reading
    # ------------------------------------------------------------------

    def _read_record(
        self,
    ) -> Optional[Tuple[str, Optional[bytes], int, bool]]:
        """Read one record from the current file cursor position.

        Returns ``None`` if the cursor is at or past the data block end.
        """
        raw_header = self._fh.read(_RECORD_HEADER_SIZE)
        if len(raw_header) < _RECORD_HEADER_SIZE:
            return None
        klen, vlen = struct.unpack(_RECORD_HEADER_FMT, raw_header)
        key = self._fh.read(klen).decode("utf-8")
        val = self._fh.read(vlen) if vlen else b""
        raw_tail = self._fh.read(_TS_DELETED_SIZE)
        ts, deleted = struct.unpack(_TS_DELETED_FMT, raw_tail)
        return key, (val if not deleted else None), ts, bool(deleted)

    def close(self) -> None:
        """Close the underlying file handle."""
        self._fh.close()

    def __repr__(self) -> str:
        return f"SSTableReader(path={os.path.basename(self.path)!r})"


# ---------------------------------------------------------------------------
# LSMLevel
# ---------------------------------------------------------------------------

class LSMLevel:
    """One level in the LSM-tree hierarchy, holding a list of SSTables.

    Reads search SSTables from newest to oldest (reverse insertion order)
    so that the most-recent version of a key is found first.

    Parameters
    ----------
    level_id:
        Integer identifier (0 = L0, 1 = L1, etc.).
    """

    def __init__(self, level_id: int) -> None:
        self.level_id = level_id
        self._tables: List[SSTableReader] = []   # ordered oldest → newest

    def add_table(self, reader: SSTableReader) -> None:
        """Register a new (newest) SSTable in this level."""
        self._tables.append(reader)

    def get(self, key: str) -> Tuple[Optional[bytes], bool]:
        """Search for *key* across all SSTables in this level.

        Parameters
        ----------
        key:
            Key to look up.

        Returns
        -------
        tuple of (value, found)
            Newest version wins.  Tombstones propagate as (None, True).
        """
        for table in reversed(self._tables):
            val, found = table.get(key)
            if found:
                return val, True
        return None, False

    def remove_tables(self, paths: List[str]) -> None:
        """Unregister SSTables by path (called after compaction)."""
        path_set = set(paths)
        self._tables = [t for t in self._tables if t.path not in path_set]

    def close_all(self) -> None:
        """Close all SSTableReader file handles in this level."""
        for t in self._tables:
            t.close()

    def __len__(self) -> int:
        return len(self._tables)

    def __repr__(self) -> str:
        return f"LSMLevel(id={self.level_id}, tables={len(self._tables)})"


# ---------------------------------------------------------------------------
# Sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import random
    import tempfile

    print("=" * 60)
    print("SSTable Sanity Check")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        sst_path = os.path.join(tmpdir, "test.sst")
        n = 1000

        # --- Write ---
        keys_vals = {f"key:{i:08d}": f"value:{i}".encode() for i in range(n)}
        with SSTableWriter(sst_path) as writer:
            for key in sorted(keys_vals):
                writer.add(key, keys_vals[key], timestamp=1)

        print(f"[PASS] Wrote {n} entries → {os.path.getsize(sst_path):,} bytes")

        # --- Read back 100 random keys ---
        reader = SSTableReader(sst_path)
        sample = random.sample(sorted(keys_vals), 100)
        for key in sample:
            val, found = reader.get(key)
            assert found, f"Key not found: {key}"
            assert val == keys_vals[key], f"Value mismatch for {key}"
        print("[PASS] 100 random reads — all values correct")

        # --- Bloom filter prevents miss scan ---
        _, found_absent = reader.get("does_not_exist_xyz")
        assert not found_absent, "False positive should be rare; key should not be found"
        print("[PASS] Absent key correctly reported as not found")

        # --- Tombstone write ---
        sst2_path = os.path.join(tmpdir, "tomb.sst")
        with SSTableWriter(sst2_path) as w2:
            w2.add("deleted_key", None, timestamp=99, is_deleted=True)
            w2.add("live_key", b"alive", timestamp=100, is_deleted=False)
        r2 = SSTableReader(sst2_path)
        val_d, found_d = r2.get("deleted_key")
        assert found_d and val_d is None, "Tombstone not recognised"
        val_l, found_l = r2.get("live_key")
        assert found_l and val_l == b"alive"
        print("[PASS] Tombstone and live key in same SSTable")

        # --- LSMLevel ---
        level = LSMLevel(level_id=0)
        level.add_table(reader)
        val_l0, found_l0 = level.get("key:00000042")
        assert found_l0
        print(f"[PASS] LSMLevel.get → {val_l0}")

        # --- BloomFilter standalone ---
        bf = BloomFilter(m=2048, k=3)
        for i in range(200):
            bf.add(f"member:{i}")
        assert all(bf.query(f"member:{i}") for i in range(200)), "Bloom false negative!"
        fps = sum(1 for i in range(200, 10_000) if bf.query(f"nonmember:{i}"))
        fpr = fps / 9800
        print(f"[PASS] BloomFilter FPR on 9800 non-members: {fpr:.3%}")

        reader.close()
        r2.close()
        level.close_all()

    print("\nAll SSTable checks passed ✓")
