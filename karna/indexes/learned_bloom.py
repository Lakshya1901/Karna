"""
karna/indexes/learned_bloom.py
===============================

Learned Bloom Filter — combining a tiny neural network with a traditional
Bloom filter to reduce memory usage while maintaining the same false-positive
rate (FPR) guarantee.

Background
----------
Kraska et al. (2018) "The Case for Learned Index Structures" introduced the
idea of replacing classical index structures with learned models.  The
**Learned Bloom Filter** (Mitzenmacher 2018; Dai et al. 2020) extends this to
probabilistic membership queries:

Traditional Bloom filter
    FPR ε requires m = -n·ln(ε) / (ln 2)² bits.  For n=1M keys and ε=1%:
    m ≈ 9.58 Mbit.

Learned Bloom filter
    A binary classifier (here: a 2-layer MLP) is trained to distinguish
    positive keys (present) from negative keys (absent).  Keys that the model
    confidently classifies as positive bypass the traditional filter entirely.
    Keys rejected by the model are checked against a much smaller *backup*
    Bloom filter sized only for the model's false-negative rate (FNR).

    Total memory ≈ model size + backup filter size
                 = O(model params) + m′   where m′ << m

Memory comparison (illustrative, n=100 000, ε=0.01):
    Traditional filter  : 9.58 Mbit
    Learned filter      : ~50 kB (MLP weights) + ~1 Mbit (backup filter)

References
----------
* Kraska et al. (2018) — The Case for Learned Index Structures. SIGMOD.
* Mitzenmacher (2018) — A Model for Learned Bloom Filters.  NeurIPS.
* Dai et al. (2020) — A Learned Bloom Filter Replacement. VLDB.
"""

from __future__ import annotations

import hashlib
import math
import struct
import sys
from typing import List, Optional

import numpy as np

# ---------------------------------------------------------------------------
# Optional sklearn import
# ---------------------------------------------------------------------------
try:
    from sklearn.neural_network import MLPClassifier  # type: ignore
    from sklearn.preprocessing import StandardScaler  # type: ignore
    _HAS_SKLEARN = True
except ImportError:  # pragma: no cover
    _HAS_SKLEARN = False


# ---------------------------------------------------------------------------
# Key featurisation
# ---------------------------------------------------------------------------

_N_STATS = 4


def _featurise(key: str, n_features: int = 32) -> List[float]:
    """Convert a string key to a fixed-length numeric feature vector.

    The model can only beat a hash-based filter if the features expose the
    key's *structure* (prefixes, digit runs, length).  Hashing the key first
    would make every key look like uniform noise and leave nothing to learn,
    so the features are the leading characters themselves plus a few shape
    statistics:

    * ``n_features - 4`` leading byte values, normalised to [0, 1] (0-padded)
    * length / 64, digit fraction, alphabetic fraction, punctuation fraction

    Parameters
    ----------
    key:
        The key to embed.
    n_features:
        Number of output features (≥ 5).

    Returns
    -------
    list of float
        Feature vector in [0, 1]^n_features.
    """
    raw = key.encode("utf-8")
    n_chars = n_features - _N_STATS
    chars = [b / 255.0 for b in raw[:n_chars]] + [0.0] * max(0, n_chars - len(raw))
    n = max(1, len(key))
    stats = [
        min(len(key), 64) / 64.0,
        sum(c.isdigit() for c in key) / n,
        sum(c.isalpha() for c in key) / n,
        sum(not c.isalnum() for c in key) / n,
    ]
    return chars + stats


# ---------------------------------------------------------------------------
# Traditional Bloom filter (for backup use)
# ---------------------------------------------------------------------------

class _TraditionalBloomFilter:
    """Minimal counting Bloom filter used as the backup filter.

    Parameters
    ----------
    n:
        Expected number of elements.
    fpr:
        Target false-positive rate (0 < fpr < 1).
    """

    def __init__(self, n: int = 1000, fpr: float = 0.01) -> None:
        self.fpr = fpr
        if n == 0:
            self.m = 128
            self.k = 3
        else:
            self.m = max(128, math.ceil(-n * math.log(fpr) / (math.log(2) ** 2)))
            self.k = max(1, round((self.m / n) * math.log(2)))
        self._bits = bytearray(math.ceil(self.m / 8))

    def _positions(self, key: str) -> List[int]:
        digest = hashlib.sha256(key.encode("utf-8")).digest()
        positions = []
        for i in range(self.k):
            start = (i * 4) % 28
            val = int.from_bytes(digest[start : start + 4], "little")
            positions.append(val % self.m)
        return positions

    def add(self, key: str) -> None:
        for pos in self._positions(key):
            self._bits[pos >> 3] |= 1 << (pos & 7)

    def query(self, key: str) -> bool:
        return all(
            (self._bits[pos >> 3] >> (pos & 7)) & 1
            for pos in self._positions(key)
        )

    @property
    def size_bytes(self) -> int:
        return len(self._bits)

    def __repr__(self) -> str:
        return f"_TraditionalBloomFilter(m={self.m}, k={self.k})"


# ---------------------------------------------------------------------------
# Pure-numpy MLP fallback
# ---------------------------------------------------------------------------

class _NumpyMLP:
    """Minimal 2-layer MLP for binary classification (sklearn-free fallback).

    Architecture: input(32) → hidden(64, ReLU) → output(1, sigmoid)

    Parameters
    ----------
    n_features:
        Number of input features.
    hidden_size:
        Hidden layer width.
    lr:
        Learning rate.
    n_epochs:
        Training epochs.
    """

    def __init__(
        self,
        n_features: int = 32,
        hidden_size: int = 64,
        lr: float = 0.01,
        n_epochs: int = 100,
    ) -> None:
        rng = np.random.default_rng(42)
        self.W1 = rng.normal(0, 0.1, (n_features, hidden_size))
        self.b1 = np.zeros(hidden_size)
        self.W2 = rng.normal(0, 0.1, (hidden_size, 1))
        self.b2 = np.zeros(1)
        self.lr = lr
        self.n_epochs = n_epochs

    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))

    def _forward(self, X: np.ndarray):
        h = np.maximum(0, X @ self.W1 + self.b1)   # ReLU
        out = self._sigmoid(h @ self.W2 + self.b2)
        return h, out

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train with mini-batch SGD."""
        batch_size = min(256, len(X))
        rng = np.random.default_rng(0)
        for _ in range(self.n_epochs):
            idx = rng.permutation(len(X))
            for start in range(0, len(X), batch_size):
                batch = idx[start : start + batch_size]
                Xb, yb = X[batch], y[batch, None]
                h, out = self._forward(Xb)
                # BCE gradient
                dout = (out - yb) / len(batch)
                dW2 = h.T @ dout
                db2 = dout.sum(0)
                dh = dout @ self.W2.T * (h > 0)
                dW1 = Xb.T @ dh
                db1 = dh.sum(0)
                self.W2 -= self.lr * dW2
                self.b2 -= self.lr * db2
                self.W1 -= self.lr * dW1
                self.b1 -= self.lr * db1

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        _, out = self._forward(X)
        return out.ravel()


# ---------------------------------------------------------------------------
# LearnedBloomFilter
# ---------------------------------------------------------------------------

class LearnedBloomFilter:
    """Learned Bloom Filter combining a neural classifier with a backup filter.

    The query path:
    1. Featurise the query key.
    2. Run the MLP classifier → probability p.
    3. If p ≥ threshold → predict "present" (may be FP, but no FN from model).
    4. If p < threshold → consult the backup Bloom filter.
       * The backup only needs to cover keys that the model misses (FN rate).

    Parameters
    ----------
    model_fpr:
        Target false-positive rate of the model stage.  After training, the
        threshold τ is set to the (1 - model_fpr) quantile of the scores of
        held-out negative keys, so the model admits about ``model_fpr`` of
        absent keys.  Overall FPR ≈ model_fpr + (1 - model_fpr) · backup_fpr.
    n_features:
        SHA-256 byte-features per key (default 32).
    backup_fpr:
        FPR target for the backup Bloom filter (catches model FNs).
    use_sklearn:
        Force use of sklearn (``True``) or numpy fallback (``False``).
        Defaults to ``None`` (auto-detect).

    Attributes
    ----------
    _model : MLPClassifier | _NumpyMLP
        Trained binary classifier.
    _backup : _TraditionalBloomFilter
        Backup filter for keys the model misclassifies as negative.
    _threshold : float
        Decision boundary probability.
    _fitted : bool
        Whether :meth:`fit` has been called.
    """

    def __init__(
        self,
        model_fpr: float = 0.005,
        n_features: int = 32,
        backup_fpr: float = 0.005,
        use_sklearn: Optional[bool] = None,
        hidden_size: int = 16,
    ) -> None:
        self._model_fpr   = model_fpr
        self._threshold   = 1.0
        self._n_features  = n_features
        self._backup_fpr  = backup_fpr
        self._fitted      = False
        self._backup: Optional[_TraditionalBloomFilter] = None

        use_sk = _HAS_SKLEARN if use_sklearn is None else (use_sklearn and _HAS_SKLEARN)
        if use_sk:
            self._model = MLPClassifier(
                hidden_layer_sizes=(hidden_size,),
                activation="relu",
                max_iter=300,
                random_state=42,
            )
            self._scaler: Optional[StandardScaler] = StandardScaler()
            self._backend = "sklearn"
        else:
            self._model = _NumpyMLP(n_features=n_features, hidden_size=hidden_size)
            self._scaler = None
            self._backend = "numpy"

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(self, positive_keys: List[str], negative_keys: List[str]) -> None:
        """Train the classifier and build the backup Bloom filter.

        Parameters
        ----------
        positive_keys:
            Keys known to be in the dataset (ground-truth positives).
        negative_keys:
            Keys known to be absent (ground-truth negatives).  Should be
            drawn from the expected query distribution.
        """
        # Hold out a third of the negatives to calibrate the threshold, so the
        # model FPR is measured on keys it was not trained on
        n_cal = max(1, len(negative_keys) // 3)
        neg_train, neg_cal = negative_keys[n_cal:], negative_keys[:n_cal]

        pos_X = np.array([_featurise(k, self._n_features) for k in positive_keys])
        neg_X = np.array([_featurise(k, self._n_features) for k in neg_train])
        X = np.vstack([pos_X, neg_X])
        y = np.concatenate([np.ones(len(positive_keys)), np.zeros(len(neg_train))])

        if self._backend == "sklearn":
            self._scaler.fit(X)
            self._model.fit(self._scaler.transform(X), y)
        else:
            self._model.fit(X, y)

        cal_scores = self._scores(neg_cal)
        self._threshold = float(np.quantile(cal_scores, 1.0 - self._model_fpr, method="higher"))
        # A score must strictly exceed τ; calibration negatives at τ stay rejected
        pos_scores = self._scores(positive_keys)

        # Positive keys the model rejects are false negatives → backup filter
        fn_keys = [k for k, p in zip(positive_keys, pos_scores) if p <= self._threshold]
        self._backup = _TraditionalBloomFilter(
            n=max(1, len(fn_keys)), fpr=self._backup_fpr
        )
        for k in fn_keys:
            self._backup.add(k)
        self._n_backup_keys = len(fn_keys)

        self._positive_keys_set = set(positive_keys)
        self._fitted = True

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def query(self, key: str) -> bool:
        """Return ``True`` if *key* might be in the set.

        Parameters
        ----------
        key:
            Query key.

        Returns
        -------
        bool
            ``True`` = possibly present (may be a false positive).
            ``False`` = definitely absent.

        Raises
        ------
        RuntimeError
            If called before :meth:`fit`.
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before query()")

        if self._scores([key])[0] > self._threshold:
            return True

        # Model says "not present" — check backup filter
        return self._backup.query(key)

    def _scores(self, keys: List[str]) -> np.ndarray:
        """Model probability that each key is present."""
        X = np.array([_featurise(k, self._n_features) for k in keys])
        if self._backend == "sklearn":
            return self._model.predict_proba(self._scaler.transform(X))[:, 1]
        return self._model.predict_proba(X)

    @property
    def backup_key_count(self) -> int:
        """Number of positive keys the model missed (stored in the backup filter)."""
        return self._n_backup_keys

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def false_positive_rate(self, test_keys: List[str]) -> float:
        """Compute empirical FPR on *test_keys* (all assumed absent).

        Parameters
        ----------
        test_keys:
            A list of keys definitely NOT in the positive set.

        Returns
        -------
        float
            Fraction of test_keys incorrectly reported as "present".
        """
        if not test_keys:
            return 0.0
        fps = sum(1 for k in test_keys if self.query(k))
        return fps / len(test_keys)

    @property
    def model_size_bytes(self) -> int:
        """Model footprint if parameters are stored as float32 (4 bytes each),
        plus the input scaler's mean/scale vectors."""
        if self._backend == "sklearn":
            params = sum(a.size for a in self._model.coefs_) + sum(a.size for a in self._model.intercepts_)
            params += 2 * self._n_features
        else:
            m: _NumpyMLP = self._model  # type: ignore
            params = m.W1.size + m.b1.size + m.W2.size + m.b2.size
        return 4 * params

    @property
    def backup_size_bytes(self) -> int:
        """Size of the backup Bloom filter in bytes."""
        return self._backup.size_bytes if self._backup else 0

    def compare_with_traditional(
        self, n_keys: int, fpr: float = 0.01
    ) -> dict:
        """Return a dict comparing learned vs traditional filter sizes.

        Parameters
        ----------
        n_keys:
            Size of the key set to compare against.
        fpr:
            Target FPR for the traditional filter.

        Returns
        -------
        dict
            Keys: 'traditional_bytes', 'learned_bytes', 'reduction_pct'.
        """
        trad = _TraditionalBloomFilter(n=n_keys, fpr=fpr)
        trad_bytes = trad.size_bytes
        learned_bytes = self.model_size_bytes + self.backup_size_bytes
        reduction = 100.0 * (1.0 - learned_bytes / max(1, trad_bytes))
        return {
            "traditional_bytes": trad_bytes,
            "learned_bytes": learned_bytes,
            "model_bytes": self.model_size_bytes,
            "backup_bytes": self.backup_size_bytes,
            "reduction_pct": reduction,
        }

    def __repr__(self) -> str:
        return (
            f"LearnedBloomFilter(backend={self._backend!r}, "
            f"threshold={self._threshold:.4f}, fitted={self._fitted})"
        )


# ---------------------------------------------------------------------------
# Sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import random

    rng = random.Random(42)
    n_pos = 20_000
    # Structured keys: the stored set is user IDs; absent lookups are other key families
    positive_keys = [f"user:{rng.randint(0, 10**9):010d}" for _ in range(n_pos)]
    negatives = [f"{rng.choice(['order', 'sess', 'cart'])}:{rng.randint(0, 10**9):010d}" for _ in range(12_000)]
    neg_train, neg_test = negatives[:6_000], negatives[6_000:]

    lbf = LearnedBloomFilter()
    lbf.fit(positive_keys, neg_train)
    fn_count = sum(1 for k in positive_keys if not lbf.query(k))
    assert fn_count == 0, "a Bloom filter must never return a false negative"
    print(f"[PASS] zero false negatives over {n_pos} stored keys")

    fpr = lbf.false_positive_rate(neg_test)
    trad = _TraditionalBloomFilter(n=n_pos, fpr=0.01)
    for k in positive_keys:
        trad.add(k)
    trad_fpr = sum(1 for k in neg_test if trad.query(k)) / len(neg_test)
    cmp = lbf.compare_with_traditional(n_keys=n_pos, fpr=0.01)
    print(f"  traditional: {trad_fpr:.3%} FPR in {cmp['traditional_bytes']:,} B")
    print(f"  learned:     {fpr:.3%} FPR in {cmp['learned_bytes']:,} B "
          f"(model {cmp['model_bytes']:,} B + backup {cmp['backup_bytes']:,} B "
          f"for {lbf.backup_key_count} missed keys)")
    assert fpr <= 0.02, f"learned filter FPR {fpr:.2%} is far above target"
    print("[PASS] learned filter FPR within target")
