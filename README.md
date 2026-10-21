# Privacy-Preserving Semantic Search Index (PSSI)

Reference implementation for **"Privacy-Preserving Semantic Search Index (PSSI) for Cloud-Based Document Retrieval"**, Md. Shamsul Arifin Mozumder and Muhammad Abdullah Adnan, *2026 IEEE International Conference on Big Data (IEEE BigData 2026)*, short paper.

PSSI supports substring, fuzzy and semantic matching over an obfuscated index held by an untrusted cloud server. The server stores only sparse integer arrays of noisy active bit indices and ranks documents with set intersection and Hamming distance. It never sees raw tokens, term frequencies, positions or dense embeddings.

## Overview

1. **Trusted client** (`pssi/client.py`, paper Sec. IV-B): normalises text, extracts character trigrams, encodes them into a Bloom filter (Eq. 3), embeds the document and projects it to a binary semantic hash (Eq. 4), then applies **bit-flip randomized-response noise** to every bit of both representations (Eq. 5) before upload.
2. **Untrusted cloud** (`pssi/cloud.py`, Sec. IV-C): stores the sparse indices and computes `P_substr`, `P_sem` (Eq. 6) and the composite score `lambda*P_substr + (1-lambda)*P_sem` (Eq. 7) with integer arithmetic only. Scoring is a linear scan over all stored entries.

The bit-flip mechanism is Warner's randomized response and satisfies **ε₀-local differential privacy** with the tight bound (Lemma 1)

```
ε₀ = ln((1 − η) / η)      →  η = 0.05 gives ε₀ = ln(19) ≈ 2.94
```

This is a *per-bit* guarantee. The paper reports it alongside a conservative fully composed document-level bound (`s·ε₀ ≈ 918` at s = 312 active bits; Theorem 2, Corollary 3) and does not claim strong formal privacy at the default setting (Remark 1).

## Configuration

Defaults match the paper (Sec. IV-D, "Default parameters"):

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

## Repository layout

* `pssi/client.py`: client-side encoding, projection and noise injection.
* `pssi/cloud.py`: sparse index storage and two-stage matching (Eqs. 6 and 7).
* `pssi/utils.py`: Bloom hashing, random projection, bit-flip noise, and the ε₀ / FPR helpers.
* `pssi/attack.py`: Algorithm 1, Sub-procedure A, the token-reconstruction attack used to measure leakage `L_A` (Sec. V).
* `verify_experiments.py`: checks the analytically reproducible claims.
* `demo.py`: end-to-end walkthrough.
* `pssi_plot_results.py`: regenerates the plots used in Figs. 2 to 4 of the paper into `Pics/`.

## Installation

Python 3.8+.

```bash
pip install -r requirements.txt
```

Core dependencies are `numpy`, `mmh3` and `matplotlib`. **`sentence-transformers` is optional**: install it to use the real `all-MiniLM-L6-v2` embeddings described in the paper. Without it the client falls back to a deterministic pseudo-embedding that exercises the pipeline but carries **no semantic signal**. That is enough for the mechanism checks below but cannot reproduce the paper's retrieval-quality numbers.

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

Checks the claims that follow from the mechanism itself: the ε₀ guarantee (Lemma 1), post-noise bit occupancy against the analytic prediction `p' = p(1−η) + (1−p)η`, the k-hash false-positive rate (Sec. IV-D), the ~87 MB FiQA-2018 index size (Sec. VI-B), monotone decay of leakage `L_A` with η (Sec. V), and the λ ablation (Sec. VI-C).

### 3. Figures

```bash
python pssi_plot_results.py
```

Regenerates the nine plots in Figs. 2 to 4 of the paper into `Pics/` from the measured values reported in the paper. Figure numbers and captions come from the paper, so the images carry no "Fig N" titles.

## Scope and reproducibility

This repository implements the **PSSI mechanism** and the leakage attack used to evaluate it. It does **not** bundle the cross-system comparison: the Elasticsearch, Lucene/Pyserini, ANCE/FAISS and OXT-SSE baselines in Table I and Figs. 2 and 4 require those external systems plus the BEIR corpora (NFCorpus, SciFact, FiQA-2018), and cannot be reproduced from this code alone.

Values printed by `verify_experiments.py` come from a small self-contained vocabulary. They match the paper **qualitatively** (trends, orders of magnitude, analytic identities) rather than reproducing the exact reported figures, which come from 500 held-out BEIR documents against a 50,000-token Wikipedia vocabulary.

The threat model is honest-but-curious (Definition 1). Active adversaries (chosen-document and chosen-query attacks), adaptive attackers, and leakage from PSSI's own access patterns and semantic hash are outside the paper's scope and are listed as future work.

## Citation

```bibtex
@inproceedings{mozumder2026pssi,
  author    = {Mozumder, Md. Shamsul Arifin and Adnan, Muhammad Abdullah},
  title     = {Privacy-Preserving Semantic Search Index ({PSSI}) for Cloud-Based Document Retrieval},
  booktitle = {Proceedings of the 2026 IEEE International Conference on Big Data (BigData)},
  year      = {2026},
  note      = {Short paper}
}
```
