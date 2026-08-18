# Privacy-Preserving Semantic Search Index (PSSI)

Prototype implementation accompanying **"Privacy-Preserving Semantic Search Index (PSSI) for Cloud-Based Document Retrieval"** (IEEE BigData 2026 submission).

PSSI enables substring, fuzzy and semantic matching over an obfuscated index held by an untrusted cloud server. The server stores only sparse integer arrays of noisy active bit indices, and matches using set intersection and Hamming distance — it never sees raw tokens, term frequencies, positions, or dense embeddings.

## Overview

1. **Trusted client** (`pssi/client.py`) — normalises text, extracts character n-grams, encodes them into a Bloom filter, embeds the document and projects it to a binary semantic hash, then applies **bit-flip randomized-response noise** to every bit of both representations before upload.
2. **Untrusted cloud** (`pssi/cloud.py`) — stores sparse indices and computes the composite ranking score using integer arithmetic only.

The bit-flip mechanism is Warner's randomized response and satisfies **ε₀-local differential privacy** with a tight

```
ε₀ = ln((1 − η) / η)      →  η = 0.05 gives ε₀ = ln(19) ≈ 2.94
```

Note this is a *per-bit* guarantee. The paper reports it honestly alongside a conservative fully-composed document-level bound (`s·ε₀ ≈ 918` at s = 312 active bits) rather than claiming a small formal ε. See Section V-C and Remark 1 of the paper.

## Configuration

Defaults match the paper (Section V-D):

| Symbol | Meaning | Default |
|---|---|---|
| `m` | Bloom filter size (bits) | 1024 |
| `k` | Hash functions | 4 |
| `n` | Character n-gram length | 3 |
| `b` | Semantic hash length (bits) | 128 |
| `p` | Embedding dimensionality | 384 (`all-MiniLM-L6-v2`) |
| `η` | Bit-flip probability | 0.05 |
| `λ` | Composite weighting | 0.5 |

Setting `eta=0.0` reproduces the **Plain BF** ablation baseline from the paper.

## Modules

* `pssi/client.py` — client-side encoding, projection and noise injection.
* `pssi/cloud.py` — sparse index storage and two-stage matching (Eqs. 8–10).
* `pssi/utils.py` — Bloom hashing, random projection, bit-flip noise, and the ε₀/FPR helpers.
* `pssi/attack.py` — Algorithm 1 Sub-procedure A, the token-reconstruction attack used to measure leakage `L_A`.

## Installation

Python 3.8+.

```bash
pip install -r requirements.txt
```

Core dependencies are `numpy`, `mmh3` and `matplotlib`. **`sentence-transformers` is optional**: install it to use real `all-MiniLM-L6-v2` embeddings as described in the paper. Without it the client falls back to a deterministic pseudo-embedding that exercises the pipeline but carries **no semantic signal** — sufficient for the mechanism checks below, but it cannot reproduce the paper's retrieval-quality numbers.

```bash
pip install sentence-transformers      # optional, for real semantic embeddings
```

## Usage

### 1. Interactive demonstration

```bash
python demo.py
```

Walks through indexing, noise injection, sparse upload, query encoding and ranked retrieval, printing the bits added and removed by the noise mechanism at each step.

### 2. Verification of analytical claims

```bash
python verify_experiments.py
```

Checks the claims that follow from the mechanism itself: the ε₀ guarantee, post-noise bit occupancy against its analytical prediction, the k-hash false-positive rate, the ~87 MB FiQA-2018 index size, monotonic decay of leakage `L_A` with η, and the λ ablation.

### 3. Figures

```bash
python pssi_plot_results.py
```

Regenerates the paper's figures into `figures/`.

## Scope and reproducibility

This repository implements the **PSSI mechanism** and the leakage attack used to evaluate it. It does **not** bundle the cross-system comparison: the Elasticsearch, Lucene/Pyserini, ANCE/FAISS and OXT-SSE baselines in Table I and Figures 1–5 and 9 require those external systems plus the BEIR corpora (NFCorpus, SciFact, FiQA-2018), and are not reproducible from this code alone.

Values printed by `verify_experiments.py` are computed from a small self-contained vocabulary and so match the paper **qualitatively** (trends, orders of magnitude, analytical identities) rather than reproducing the exact reported figures, which come from 500 held-out BEIR documents against a 50,000-token Wikipedia vocabulary.

## Citation

Citation details to follow upon acceptance.
