# Karna

An LSM-tree key-value store in Python with a **learned Bloom filter**: a small neural classifier in front of a backup Bloom filter, after Kraska et al. (2018) and Mitzenmacher (2018).

```
put(k, v) ─► WAL (append, crash recovery) ─► MemTable (sorted) ──flush──► L0 SSTable
                                                                            │  ≥ 4 files
                                                                            ▼
get(k) ─► MemTable ─► L0 (newest first) ─► L1         merged, newest version wins ─► L1 SSTable
                      └── per-SSTable Bloom filter skips tables that can't hold k
```

| Module | What it does |
|---|---|
| `karna/storage/memtable.py` | sorted in-memory buffer with tombstones and a size-based flush threshold |
| `karna/storage/sstable.py` | immutable sorted files: data block, sparse index, Bloom filter, footer + magic |
| `karna/storage/lsm_engine.py` | put/get/delete, WAL replay, flush, L0→L1 compaction |
| `karna/indexes/learned_bloom.py` | learned Bloom filter (MLP + calibrated threshold + backup filter) |
| `karna/benchmarks/benchmark.py` | throughput, latency and filter FPR/memory benchmarks |

## The learned Bloom filter

A classic Bloom filter needs about `-n·ln(ε)/(ln 2)²` bits regardless of what the keys look like. A learned filter trains a classifier to tell stored keys from the keys you'll actually query. It then sets the decision threshold on **held-out** negative keys so the model admits a target fraction ε_model of absent keys. Every stored key the model rejects goes into a small backup filter, so there are **never false negatives**:

    FPR ≈ ε_model + (1 − ε_model) · ε_backup

Keys are featurised from their characters and shape (leading bytes, length, digit/alpha/punctuation ratios), not from a hash. Hashing first would turn every key into uniform noise and leave the model nothing to learn.

### Measured results

20,000 stored keys (`user:<10 digits>`), 10,000 absent test keys, both filters targeting about 1% FPR (`python3 karna/benchmarks/benchmark.py`):

| Absent keys look like… | Classic filter | Learned filter |
|---|---|---|
| other key families (`order:`, `sess:`, `cart:`) | 0.97% FPR, 23,963 B | **0.48% FPR, 2,452 B** (10× smaller) |
| stored keys (`user:`, same shape) | 0.88% FPR, 23,963 B | 1.01% FPR, 29,868 B (25% larger) |

The trade-off is visible in both rows. When absent keys have learnable structure, the model does most of the filtering and the backup filter is nearly empty. When they don't, calibration correctly refuses to trust the model: almost every key lands in the backup filter, the FPR guarantee still holds, and you pay for the model on top. Queries are also slower in this Python implementation (about 150–240 µs vs 6–8 µs), because each lookup runs an MLP forward pass.

Engine throughput on the same machine (single process, CPython): about 67k writes/s (p99 0.12 ms) and about 2.5M MemTable-hit reads/s.

## Bugs fixed while validating

- **SSTable Bloom filters were a fixed 1,024 bits.** Any table with more than a few hundred keys saturated its filter, so every lookup read every table. Filters are now sized from each table's key count at the target FPR.
- **Hash positions repeated for k > 3.** Bit positions came from fixed 8-byte digest slices that cycled after three. They now use Kirsch–Mitzenmacher double hashing.
- **The learned filter had a 48% FPR.** It featurised SHA-256 digests and used a fixed 0.5 threshold. It now uses structural features and threshold calibration, as described above.

## Running

```bash
pip install -r requirements.txt
pytest -q                                  # 13 tests
python3 karna/benchmarks/benchmark.py      # numbers above
python3 karna/storage/lsm_engine.py        # per-module sanity checks also run standalone
```

The tests cover reads across flushes and repeated compactions (newest version wins across L1 tables), tombstones shadowing values in lower levels, WAL crash recovery, clean restarts, SSTable round-trips, Bloom filter FPR, and the learned filter's no-false-negative guarantee on both key distributions.

## Status and next steps

- `karna-rust/` is an **uncompiled skeleton** of a Rust port (storage crate drafted, server/client/consensus crates empty). It is not part of the working system.
- Compaction is a single L0→L1 merge. Leveled compaction with per-level size targets, and dropping tombstones at the bottom level, are the next storage steps.
- Nothing is distributed yet. Replication (Raft) would sit above `LSMEngine`.
