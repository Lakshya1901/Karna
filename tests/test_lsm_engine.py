import os

import pytest

from karna.storage.lsm_engine import LSMEngine
from karna.storage.memtable import MemTable
from karna.storage.sstable import BloomFilter, SSTableReader, SSTableWriter

TINY = 0.001  # ~1 KiB memtable, so a few dozen writes force a flush


@pytest.fixture
def engine(tmp_path):
    e = LSMEngine(str(tmp_path), memtable_threshold_mb=TINY)
    yield e
    e.close()


def test_put_get_delete(engine):
    engine.put("a", b"1")
    engine.put("a", b"2")
    assert engine.get("a") == b"2"
    engine.delete("a")
    assert engine.get("a") is None
    assert engine.get("never") is None


def test_reads_survive_flushes_and_compaction(engine):
    expected = {f"k{i:05d}": f"v{i}".encode() for i in range(2_000)}
    for k, v in expected.items():
        engine.put(k, v)
    assert len(engine._l1) >= 1, "workload should have triggered L0→L1 compaction"
    assert all(engine.get(k) == v for k, v in expected.items())


def test_latest_write_wins_across_repeated_compactions(engine):
    for round_ in range(3):
        for i in range(1_000):
            engine.put(f"k{i:04d}", f"r{round_}".encode())
        engine.put("hot", f"r{round_}".encode())
    assert len(engine._l1) >= 2, "need several L1 tables to exercise cross-table ordering"
    assert engine.get("hot") == b"r2"
    assert engine.get("k0500") == b"r2"


def test_delete_shadows_older_value_in_lower_level(engine):
    engine.put("gone", b"old")
    for i in range(2_000):
        engine.put(f"fill{i:05d}", b"x")  # push "gone" down to L1
    engine.delete("gone")
    assert engine.get("gone") is None
    for i in range(2_000):
        engine.put(f"more{i:05d}", b"x")  # push the tombstone down too
    assert engine.get("gone") is None


def test_wal_recovers_unflushed_writes_after_crash(tmp_path):
    e = LSMEngine(str(tmp_path), memtable_threshold_mb=4)
    e.put("durable", b"yes")
    e.delete("tmp")
    e._wal.flush()
    # Simulate a crash: drop the engine without close() (no memtable flush)
    e._wal.close()
    reopened = LSMEngine(str(tmp_path), memtable_threshold_mb=4)
    assert reopened.get("durable") == b"yes"
    reopened.close()


def test_data_persists_across_clean_restart(tmp_path):
    e = LSMEngine(str(tmp_path), memtable_threshold_mb=TINY)
    for i in range(1_500):
        e.put(f"k{i}", str(i).encode())
    e.close()
    e = LSMEngine(str(tmp_path), memtable_threshold_mb=TINY)
    assert e.get("k0") == b"0" and e.get("k1499") == b"1499"
    e.close()


def test_memtable_tracks_tombstones_and_order():
    m = MemTable(flush_threshold_bytes=1 << 20)
    for k in ("c", "a", "b"):
        m.put(k, k.encode())
    m.delete("b")
    assert m.get("b") is None and not m.contains("b")
    # The tombstone is still emitted on flush so it can shadow older SSTable values
    rows = list(m.to_sorted_iterator())
    assert [k for k, *_ in rows] == ["a", "b", "c"]
    assert [deleted for *_, deleted in rows] == [False, True, False]


def test_sstable_round_trip_and_bloom(tmp_path):
    path = os.path.join(tmp_path, "t.sst")
    w = SSTableWriter(path)
    for i in range(500):
        w.add(f"key{i:04d}", f"val{i}".encode(), i, False)
    w.add("key9999", None, 999, True)
    w.finish()
    r = SSTableReader(path)
    assert r.get("key0123") == (b"val123", True)
    assert r.get("key9999") == (None, True)       # tombstone is "found"
    assert r.get("absent")[1] is False
    r.close()


def test_bloom_filter_has_no_false_negatives_and_low_fpr():
    bf = BloomFilter.for_capacity(5_000, fpr=0.01)
    keys = [f"in{i}" for i in range(5_000)]
    for k in keys:
        bf.add(k)
    assert all(bf.query(k) for k in keys)
    fpr = sum(bf.query(f"out{i}") for i in range(20_000)) / 20_000
    assert fpr < 0.02
