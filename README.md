# Privacy-Preserving Semantic Search Index (PSSI)

Source code for the paper

> **Privacy-Preserving Semantic Search Index (PSSI) for Cloud-Based Document Retrieval**
> Md. Shamsul Arifin Mozumder and Muhammad Abdullah Adnan
> 2026 IEEE International Conference on Big Data (IEEE BigData 2026), short paper

Repository: <https://github.com/arifincsebuet/PSSI>

PSSI lets an untrusted cloud server answer substring, fuzzy, and semantic queries without seeing the documents. The server stores only sparse integer arrays of noisy active-bit indices and ranks documents with set intersection and Hamming distance. Raw tokens, term frequencies, positions, and dense embeddings never leave the client.

## How it works

1. **Trusted client** (`pssi/client.py`, paper Sec. IV-B)
   - Normalizes the text and extracts character trigrams.
   - Encodes the trigrams into a Bloom filter (Eq. 3).
   - Embeds the text and projects it to a 128-bit semantic hash (Eq. 4).
   - Flips every bit of both representations with probability η (Eq. 5).
   - Uploads only the indices of the active bits.
2. **Untrusted cloud** (`pssi/cloud.py`, Sec. IV-C)
   - Stores the sparse index entries.
   - Computes the Bloom overlap `P_substr` and the Hamming similarity `P_sem` (Eq. 6).
   - Ranks by `λ·P_substr + (1−λ)·P_sem` (Eq. 7) and returns the top-k document IDs.
   - Scoring is a linear scan over all stored entries, using integer operations only.

## Privacy guarantee

The bit flip is Warner's randomized response. Each bit satisfies ε₀-local differential privacy with the tight bound of Lemma 1 (Eq. 8):

```
ε₀ = ln((1 − η) / η)        η = 0.05  →  ε₀ = ln 19 ≈ 2.94
```

This is a per-bit guarantee. The paper also reports (Corollary 3):

| Accounting | Bound at η = 0.05 |
|---|---|
| Per bit | ε₀ ≈ 2.94 |
| Per n-gram (2k bits, k = 4) | 2k·ε₀ ≈ 23.6 |
| Whole document (m + b = 1,152 bits) | (m + b)·ε₀ ≈ 3,392 |

The paper does not claim strong formal privacy at the default setting (Remark 1). η is a tunable knob, and the attack evaluation is the main evidence of protection.

## Default configuration

Defaults match the paper (Sec. IV-D, "Default parameters").

| Symbol | Meaning | Default |
|---|---|---|
| `m` | Bloom filter size (bits) | 1024 |
| `k` | Hash functions | 4 |
| `n` | Character n-gram length | 3 |
| `b` | Semantic hash length (bits) | 128 |
| `p` | Embedding dimensionality | 384 (`all-MiniLM-L6-v2`) |
| `η` | Bit-flip probability | 0.05 |
| `λ` | Composite weighting | 0.5 |

Setting `eta=0.0` and `lam=1.0` gives the **Plain BF** baseline from the paper.

## Repository layout

| Path | Contents | Paper |
|---|---|---|
| `pssi/client.py` | Client-side encoding, projection, noise injection | Sec. IV-B |
| `pssi/cloud.py` | Sparse index store and two-stage scoring | Sec. IV-C |
| `pssi/utils.py` | Bloom hashing, random projection, bit-flip noise, ε₀ and FPR helpers | Eqs. 3 to 5, 8 |
| `pssi/attack.py` | Token-reconstruction attack (Algorithm 1, Sub-procedure A) | Sec. V |
| `demo.py` | End-to-end walkthrough | |
| `verify_experiments.py` | Checks the analytically reproducible claims | Sec. VI-E |
| `pssi_plot_results.py` | Redraws the plots of Figs. 2 to 4 into `Pics/` | Sec. VI |
| `Pics/` | The nine plot images used in the paper | Figs. 2 to 4 |

## Installation

Requires Python 3.8 or later.

```bash
pip install -r requirements.txt
```

Core dependencies are `numpy`, `mmh3`, and `matplotlib`.

`sentence-transformers` is optional. Install it to use the real `all-MiniLM-L6-v2` embeddings described in the paper:

```bash
pip install sentence-transformers
```

Without it, the client falls back to a deterministic pseudo-embedding. That exercises the full pipeline but carries no semantic signal, so it cannot reproduce retrieval-quality numbers.

## Usage

### 1. Demonstration

```bash
python demo.py
```

Walks through indexing, noise injection, sparse upload, query encoding, and ranked retrieval. It prints the bits added and removed by the noise at each step.

### 2. Verification of analytical claims

```bash
python verify_experiments.py
```

Checks the claims that follow from the mechanism itself:

- the per-bit ε₀, the per-n-gram bound, and the composed document bound (Lemma 1, Corollary 3)
- post-noise bit occupancy against `p' = p(1−η) + (1−p)η` (Sec. IV-D)
- the k-hash false-positive rate (Sec. IV-D)
- the ≈87 MB FiQA-2018 index size (Sec. VI-B)
- monotone decay of the leakage `L_A` as η grows (Sec. V)
- the λ ablation trend (Sec. VI-C)

The script uses a small built-in vocabulary. Its outputs match the paper qualitatively (trends, orders of magnitude, analytic identities). They are not the paper's reported figures, which come from 500 held-out BEIR documents and a 50,000-token Wikipedia vocabulary.

### 3. Plots

```bash
python pssi_plot_results.py
```

Redraws the nine plots of Figs. 2 to 4 into `Pics/`. The values are hard-coded in the script and are the ones reported in the paper; the script does not rerun any experiment. Figure numbers and captions come from the paper, so the images carry no "Fig N" titles.

## Scope and reproducibility

This repository contains the **PSSI mechanism** and the token-reconstruction attack used to evaluate it.

It does not contain:

- the baselines of Table I: Elasticsearch, Lucene/Pyserini, ANCE/FAISS, and OXT SSE
- the access-pattern attack on OXT (Algorithm 1, Sub-procedure B)
- the BEIR corpora (NFCorpus, SciFact, FiQA-2018)
- the evaluation harness that produced Table I and Figs. 2 to 4

Reproducing the cross-system results therefore requires those external systems and datasets in addition to this code.

## Threat model

The server is honest-but-curious (Definition 1). It sees every stored index entry, every query payload, and the Bloom hash functions, but not the documents, the projection matrix, or the embeddings.

Outside the paper's scope and left as future work (Sec. VII):

- active adversaries, including chosen-document and chosen-query attacks
- adaptive attackers that account for the known η
- leakage from PSSI's own access patterns and from the semantic hash

## Citation

```bibtex
@inproceedings{mozumder2026pssi,
  author    = {Mozumder, Md. Shamsul Arifin and Adnan, Muhammad Abdullah},
  title     = {Privacy-Preserving Semantic Search Index ({PSSI}) for Cloud-Based Document Retrieval},
  booktitle = {Proc. 2026 IEEE International Conference on Big Data (BigData)},
  year      = {2026},
  note      = {Short paper}
}
```

## Acknowledgments

This research was supported by Samsung R&D Institute Bangladesh Ltd. (SRBD).
