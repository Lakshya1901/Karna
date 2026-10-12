# Karna — LSM-Tree Key-Value Store (Python implementation)

> **Status:** Research prototype (Phase 4). Production Rust implementation planned for Phase 6.

Karna is a research-quality Python implementation of an **LSM-tree (Log-Structured Merge-Tree)** key-value store, designed to:
1. Validate algorithmic correctness before a Rust port.
2. Serve as a citable artefact in the associated paper.
3. Provide a benchmark baseline for the learned bloom filter innovation.

---

## Architecture Overview

```
Write Path
──────────
  put(key, value)
       │
       ▼
  WAL (wal.log) ──────────────────► crash recovery
       │
       ▼
  MemTable (SortedDict / RAM)
       │  (size ≥ flush_threshold)
       ▼
  L0 SSTable file  (immutable, sorted)
       │  (L0 count ≥ 4)
       ▼
  L1 SSTable file  (merged, sorted, non-overlapping)

Read Path
─────────
  get(key)
   1. MemTable    ──► hit? return
   2. L0 (newest→oldest)  ──► bloom check, binary index search, scan
   3. L1 (newest→oldest)  ──► same
   4. return None (key absent)
```

### Components

| File | Role |
|---|---|
| `storage/memtable.py` | In-memory sorted write buffer using `SortedDict` (skip-list analogue) |
| `storage/sstable.py` | On-disk immutable sorted table with sparse index and bloom filter |
| `storage/lsm_engine.py` | Orchestrator: WAL, flush, compaction, multi-level read |
| `indexes/learned_bloom.py` | Learned Bloom Filter (neural network + backup filter) |
| `benchmarks/benchmark.py` | Throughput, latency, and filter comparison benchmarks |

---

## The Learned Bloom Filter Innovation

Traditional Bloom filters allocate bits proportional to `−n·ln(ε) / (ln 2)²`.
For 1 million keys at 1% FPR this requires **≈ 9.58 Mbit**.

Our **Learned Bloom Filter** (based on Kraska et al. 2018, Mitzenmacher 2018):

```
Query(key):
  prob = MLP(featurise(key))
  if prob ≥ threshold:
      return "possibly present"   ← model handles majority of positives
  else:
      return backup_bloom.query(key)   ← catches model false negatives
```

The backup filter is sized only for keys the model misclassifies (~5–15% of positive keys),
dramatically shrinking its required capacity:

| Filter | Memory (n=100k, ε=1%) |
|---|---|
| Traditional | ~960 KB |
| Learned (model + backup) | ~50–150 KB |
| **Reduction** | **~85%** |

---

## Performance Characteristics

### Write Amplification
- **Writes hit the WAL once + MemTable in-place** → WA ≈ 1× initially.
- Flush to L0: WA += 1× (sequential write).
- L0→L1 compaction: WA += merge cost (typically 10–30× for production systems).

### Read Amplification
- **Best case** (MemTable hit): 1 comparison.
- **Worst case** (key in L1): bloom-pass + binary-search × (L0 + L1 files).
- Bloom filters reduce worst-case I/O dramatically (typical FPR ≈ 1%).

### Space Amplification
- Size-tiered compaction (current): SA ≈ 2× (old + new files coexist during compaction).
- Level-based compaction (future Rust): SA ≈ 1.1×.

---

## Connection to Real Systems

| System | MemTable | Compaction | Notes |
|---|---|---|---|
| LevelDB | Skip list | Levelled | Google, ~2011 |
| RocksDB | Skip list / HashLinkedList | Levelled + FIFO | Facebook fork |
| Cassandra | Skip list | Size-tiered | Distributed, AP |
| **Karna** | SortedDict (Python) | Size-tiered (prototype) | Research |

Key differences from production systems:
- No multi-threading / concurrent writes (single-threaded WAL lock in Rust target).
- No block cache (OS page cache is relied upon).
- Bloom filter per SSTable, not per level (matching LevelDB design).

---

## Future Rust Implementation Plan

Phase 6 implementation targets:

1. **MemTable** — `crossbeam-skiplist::SkipMap` for lock-free concurrent reads/writes.
2. **SSTable** — mmap-backed files with `pread()` for direct block access.
3. **WAL** — `io_uring`-based async appends.
4. **Compaction** — Background tokio task, priority queue over L0 sizes.
5. **Bloom filter** — Implement `LearnedBloomFilter` in Rust using `tch-rs` (LibTorch bindings).
6. **Networking** — Raft consensus via `openraft` for distributed coordination.

---

## Running the Benchmarks

```bash
# From the Karna project root
cd /path/to/Karna

# Run individual module sanity checks
python -m karna.storage.memtable
python -m karna.storage.sstable
python -m karna.storage.lsm_engine
python -m karna.indexes.learned_bloom

# Full benchmark suite
python -m karna.benchmarks.benchmark
```

Install optional dependencies for best experience:

```bash
pip install sortedcontainers scikit-learn numpy pandas matplotlib
```

---

## References

1. O'Neil et al. (1996). *The Log-Structured Merge-Tree (LSM-Tree)*. Acta Informatica.
2. Pugh (1990). *Skip Lists: A Probabilistic Alternative to Balanced Trees*. CACM.
3. Chang et al. (2006). *Bigtable: A Distributed Storage System for Structured Data*. OSDI.
4. Kraska et al. (2018). *The Case for Learned Index Structures*. SIGMOD.
5. Mitzenmacher (2018). *A Model for Learned Bloom Filters and Optimizing by Sandwiching*. NeurIPS.
6. Dai et al. (2020). *A Learned Bloom Filter Replacement via False Positive Rate Reduction*. VLDB.
7. Hahn et al. (2020). *Dostoevsky: Better Space-Time Trade-Offs for LSM-Tree Based Key-Value Stores*. SIGMOD.
