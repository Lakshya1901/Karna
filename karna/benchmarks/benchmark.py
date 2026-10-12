"""
karna/benchmarks/benchmark.py
==============================

Performance benchmarking suite for the Karna LSM-tree storage engine.

Metrics collected
-----------------
* Write throughput (ops/sec), p50/p95/p99 write latency
* Read throughput (ops/sec) with configurable hit rate
* Bloom filter comparison: FPR, memory, and query latency

All results are collected into a :class:`pandas.DataFrame` for easy
inspection and export.  ``plot_benchmarks`` renders a two-panel matplotlib
figure (throughput vs. dataset size, latency percentiles).

Usage::

    python -m karna.benchmarks.benchmark

Or from Python::

    suite = BenchmarkSuite()
    df = suite.run_all(data_dir='/tmp/karna_bench')
    plot_benchmarks(df, '/tmp/karna_bench/results.png')
"""

from __future__ import annotations

import os
import random
import sys
import tempfile
import time
from typing import Dict, List, Optional

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from karna.storage.lsm_engine import LSMEngine
from karna.indexes.learned_bloom import LearnedBloomFilter, _TraditionalBloomFilter

# Optional heavy dependencies
try:
    import pandas as pd  # type: ignore
    _HAS_PANDAS = True
except ImportError:
    _HAS_PANDAS = False

try:
    import matplotlib.pyplot as plt  # type: ignore
    import matplotlib
    matplotlib.use("Agg")
    _HAS_MATPLOTLIB = True
except ImportError:
    _HAS_MATPLOTLIB = False


# ---------------------------------------------------------------------------
# Latency helpers
# ---------------------------------------------------------------------------

def _percentile(data: List[float], pct: float) -> float:
    """Return the *pct*-th percentile of *data* (0–100)."""
    if not data:
        return float("nan")
    sorted_data = sorted(data)
    idx = max(0, int(len(sorted_data) * pct / 100) - 1)
    return sorted_data[min(idx, len(sorted_data) - 1)]


# ---------------------------------------------------------------------------
# BenchmarkSuite
# ---------------------------------------------------------------------------

class BenchmarkSuite:
    """Collection of micro-benchmarks for Karna storage engine.

    Each benchmark returns a ``dict`` with metric names as keys.  These
    dicts are assembled into a :class:`pandas.DataFrame` by :meth:`run_all`.

    Parameters
    ----------
    random_seed:
        Seed for reproducible key generation.
    """

    def __init__(self, random_seed: int = 42) -> None:
        self._rng = random.Random(random_seed)
        self._np_rng = np.random.default_rng(random_seed)

    # ------------------------------------------------------------------
    # Write throughput
    # ------------------------------------------------------------------

    def benchmark_write_throughput(
        self,
        engine: LSMEngine,
        n_ops: int = 10_000,
    ) -> Dict:
        """Measure sequential write throughput.

        Writes *n_ops* unique key-value pairs with random 64-byte values,
        records per-operation latency, and computes aggregate throughput.

        Parameters
        ----------
        engine:
            Pre-initialised :class:`LSMEngine` instance.
        n_ops:
            Number of write operations to perform.

        Returns
        -------
        dict
            Keys: ``ops_sec``, ``p50_ms``, ``p95_ms``, ``p99_ms``,
            ``total_sec``, ``n_ops``.
        """
        keys = [f"wbench:{i:010d}" for i in range(n_ops)]
        value = b"x" * 64

        latencies: List[float] = []
        t_start = time.perf_counter()
        for key in keys:
            t0 = time.perf_counter()
            engine.put(key, value)
            latencies.append((time.perf_counter() - t0) * 1000)  # ms
        total_sec = time.perf_counter() - t_start

        return {
            "benchmark": "write_throughput",
            "n_ops": n_ops,
            "ops_sec": n_ops / total_sec,
            "total_sec": round(total_sec, 4),
            "p50_ms": round(_percentile(latencies, 50), 4),
            "p95_ms": round(_percentile(latencies, 95), 4),
            "p99_ms": round(_percentile(latencies, 99), 4),
        }

    # ------------------------------------------------------------------
    # Read throughput
    # ------------------------------------------------------------------

    def benchmark_read_throughput(
        self,
        engine: LSMEngine,
        n_ops: int = 5_000,
        hit_rate: float = 0.8,
    ) -> Dict:
        """Measure random read throughput with a configurable hit/miss ratio.

        First populates the engine with *n_ops* keys.  Then issues reads where
        ``hit_rate`` fraction target existing keys and the remainder target
        absent keys.

        Parameters
        ----------
        engine:
            :class:`LSMEngine` instance (may already contain data).
        n_ops:
            Number of read operations to issue.
        hit_rate:
            Fraction of reads that should hit an existing key (0–1).

        Returns
        -------
        dict
            Keys: ``ops_sec``, ``p50_ms``, ``p95_ms``, ``p99_ms``,
            ``empirical_hit_rate``, ``n_ops``.
        """
        present_keys = [f"rbench:{i:010d}" for i in range(n_ops)]
        absent_keys  = [f"absent:{i:010d}" for i in range(n_ops)]
        for k in present_keys:
            engine.put(k, b"v" * 32)

        n_hits  = int(n_ops * hit_rate)
        n_misses = n_ops - n_hits
        query_keys = (
            self._rng.choices(present_keys, k=n_hits)
            + self._rng.choices(absent_keys, k=n_misses)
        )
        self._rng.shuffle(query_keys)

        latencies: List[float] = []
        actual_hits = 0
        t_start = time.perf_counter()
        for key in query_keys:
            t0 = time.perf_counter()
            val = engine.get(key)
            latencies.append((time.perf_counter() - t0) * 1000)
            if val is not None:
                actual_hits += 1
        total_sec = time.perf_counter() - t_start

        return {
            "benchmark": "read_throughput",
            "n_ops": n_ops,
            "ops_sec": n_ops / total_sec,
            "total_sec": round(total_sec, 4),
            "p50_ms": round(_percentile(latencies, 50), 4),
            "p95_ms": round(_percentile(latencies, 95), 4),
            "p99_ms": round(_percentile(latencies, 99), 4),
            "empirical_hit_rate": round(actual_hits / n_ops, 4),
        }

    # ------------------------------------------------------------------
    # Bloom filter comparison
    # ------------------------------------------------------------------

    def benchmark_bloom_filter(
        self,
        traditional: _TraditionalBloomFilter,
        learned: LearnedBloomFilter,
        test_keys: List[str],
        positive_keys: Optional[List[str]] = None,
    ) -> Dict:
        """Compare traditional vs learned bloom filter.

        Parameters
        ----------
        traditional:
            A fitted :class:`_TraditionalBloomFilter`.
        learned:
            A fitted :class:`LearnedBloomFilter`.
        test_keys:
            Keys definitely absent — used to measure empirical FPR.
        positive_keys:
            Keys definitely present — used to verify zero FNs.

        Returns
        -------
        dict
            Keys: ``traditional_fpr``, ``learned_fpr``,
            ``traditional_bytes``, ``learned_bytes``,
            ``trad_query_us``, ``learned_query_us``.
        """
        n_test = len(test_keys)

        # Traditional FPR
        t0 = time.perf_counter()
        trad_fps = sum(1 for k in test_keys if traditional.query(k))
        trad_query_sec = (time.perf_counter() - t0) / max(n_test, 1)

        # Learned FPR
        t0 = time.perf_counter()
        lbf_fps = sum(1 for k in test_keys if learned.query(k))
        lbf_query_sec = (time.perf_counter() - t0) / max(n_test, 1)

        return {
            "benchmark": "bloom_filter",
            "traditional_fpr": round(trad_fps / max(n_test, 1), 5),
            "learned_fpr": round(lbf_fps / max(n_test, 1), 5),
            "traditional_bytes": traditional.size_bytes,
            "learned_bytes": learned.model_size_bytes + learned.backup_size_bytes,
            "trad_query_us": round(trad_query_sec * 1e6, 3),
            "learned_query_us": round(lbf_query_sec * 1e6, 3),
            "n_test_keys": n_test,
        }

    # ------------------------------------------------------------------
    # run_all
    # ------------------------------------------------------------------

    def run_all(self, data_dir: str = "/tmp/karna_bench") -> "pd.DataFrame | list":
        """Run the full benchmark suite and return results as a DataFrame.

        Parameters
        ----------
        data_dir:
            Directory for engine data.  Created if absent.

        Returns
        -------
        pandas.DataFrame or list of dict
            One row per benchmark.  If pandas is unavailable, returns a list
            of dicts instead.
        """
        os.makedirs(data_dir, exist_ok=True)
        results = []

        # ---- Write throughput ----
        print("  Running write throughput benchmark…")
        with tempfile.TemporaryDirectory(dir=data_dir) as wdir:
            engine = LSMEngine(wdir, memtable_threshold_mb=1.0)
            res = self.benchmark_write_throughput(engine, n_ops=5_000)
            engine.close()
        print(f"    {res['ops_sec']:.0f} ops/sec  (p99={res['p99_ms']:.3f} ms)")
        results.append(res)

        # ---- Read throughput ----
        print("  Running read throughput benchmark…")
        with tempfile.TemporaryDirectory(dir=data_dir) as rdir:
            engine = LSMEngine(rdir, memtable_threshold_mb=1.0)
            res = self.benchmark_read_throughput(engine, n_ops=2_000, hit_rate=0.8)
            engine.close()
        print(f"    {res['ops_sec']:.0f} ops/sec  hit_rate={res['empirical_hit_rate']:.2%}")
        results.append(res)

        # ---- Bloom filter: learnable vs unlearnable negative keys ----
        print("  Running bloom filter benchmark…")
        rng = random.Random(99)
        n_pos = 20_000
        pos_keys = [f"user:{rng.randint(0, 10**9):010d}" for _ in range(n_pos)]
        trad = _TraditionalBloomFilter(n=n_pos, fpr=0.01)
        for k in pos_keys:
            trad.add(k)

        scenarios = {
            # absent lookups come from other key families → structure the model can learn
            "structured": ["order", "sess", "cart"],
            # absent lookups look exactly like stored keys → nothing to learn
            "unstructured": ["user"],
        }
        for name, prefixes in scenarios.items():
            def neg(n: int) -> List[str]:
                return [f"{rng.choice(prefixes)}:{rng.randint(0, 10**9):010d}" for _ in range(n)]
            lbf = LearnedBloomFilter()
            lbf.fit(pos_keys, neg(6_000))
            res = self.benchmark_bloom_filter(trad, lbf, neg(10_000), pos_keys)
            res["benchmark"] = f"bloom_{name}"
            print(f"    [{name}] Traditional FPR={res['traditional_fpr']:.3%}  "
                  f"Learned FPR={res['learned_fpr']:.3%}  "
                  f"Memory: {res['traditional_bytes']:,}B → {res['learned_bytes']:,}B")
            results.append(res)

        if _HAS_PANDAS:
            return pd.DataFrame(results)
        return results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_benchmarks(
    results: "pd.DataFrame | list",
    save_path: str,
) -> None:
    """Generate a two-panel benchmark figure and save to *save_path*.

    Panel 1: Bar chart of throughput (ops/sec) for write and read benchmarks.
    Panel 2: Bar chart of p50/p95/p99 latencies.

    Parameters
    ----------
    results:
        Output of :meth:`BenchmarkSuite.run_all`.
    save_path:
        Path to save the PNG figure.
    """
    if not _HAS_MATPLOTLIB:
        print("[WARN] matplotlib not available — skipping plot")
        return

    if _HAS_PANDAS and hasattr(results, "to_dict"):
        rows = results.to_dict("records")
    else:
        rows = results

    # Filter to throughput rows
    tput_rows = [r for r in rows if r.get("benchmark") in ("write_throughput", "read_throughput")]
    bloom_row = next((r for r in rows if r.get("benchmark") == "bloom_filter"), None)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Karna LSM Engine — Benchmark Results", fontsize=14, fontweight="bold")

    # ---- Throughput ----
    ax1 = axes[0]
    labels = [r["benchmark"].replace("_throughput", "") for r in tput_rows]
    ops_secs = [r["ops_sec"] for r in tput_rows]
    bars = ax1.bar(labels, ops_secs, color=["#4C72B0", "#DD8452"])
    ax1.set_ylabel("Throughput (ops/sec)")
    ax1.set_title("Write vs Read Throughput")
    ax1.set_ylim(0, max(ops_secs) * 1.3)
    for bar, val in zip(bars, ops_secs):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(ops_secs) * 0.02,
            f"{val:,.0f}",
            ha="center", va="bottom", fontsize=10,
        )

    # ---- Latency percentiles ----
    ax2 = axes[1]
    if tput_rows:
        pcts = ["p50_ms", "p95_ms", "p99_ms"]
        x = np.arange(len(pcts))
        width = 0.35
        for i, row in enumerate(tput_rows):
            vals = [row.get(p, 0) for p in pcts]
            ax2.bar(x + i * width, vals, width, label=row["benchmark"].replace("_throughput", ""))
        ax2.set_xticks(x + width / 2)
        ax2.set_xticklabels(["P50", "P95", "P99"])
        ax2.set_ylabel("Latency (ms)")
        ax2.set_title("Latency Percentiles")
        ax2.legend()

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"[INFO] Benchmark figure saved → {save_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Sanity check / main entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("Karna Benchmark Suite")
    print(f"  pandas     : {_HAS_PANDAS}")
    print(f"  matplotlib : {_HAS_MATPLOTLIB}")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as bench_dir:
        suite = BenchmarkSuite(random_seed=42)
        results = suite.run_all(data_dir=bench_dir)

        print("\n--- Results ---")
        if _HAS_PANDAS:
            import pandas as pd
            pd.set_option("display.max_columns", None)
            pd.set_option("display.width", 120)
            print(results.to_string(index=False))
        else:
            for row in results:
                print(row)

        # Save plot if matplotlib available
        plot_path = os.path.join(bench_dir, "benchmark_results.png")
        plot_benchmarks(results, save_path=plot_path)
        if os.path.exists(plot_path):
            print(f"\n[PASS] Plot written to {plot_path}")
        else:
            print("\n[INFO] Plot skipped (matplotlib unavailable)")

    print("\nBenchmark suite completed ✓")
