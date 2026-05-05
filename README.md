# Privacy-Preserving Semantic Search Index (PSSI)

This repository contains the prototype implementation of the **Privacy-Preserving Semantic Search Index (PSSI)**, which accompanies the research paper: **"Privacy-Preserving Semantic Search Index (PSSI) for Cloud-Based Document Retrieval"**. 

PSSI is an architecture designed to enable secure and privacy-preserving substring and semantic token matching over encrypted data on an untrusted cloud server.

## Overview

The PSSI architecture strictly limits processing bounds by operating in two distinct spheres:
1. **Trusted Client**: Responsible for extracting n-grams/tokens from documents and queries, and building obfuscated sparse representations using specific hashing techniques (such as Bloom Filters) and random-projection binary semantic hashing. It introduces a calibrated bit-flip noise mechanism with formal $(\epsilon,\delta)$-differential privacy bounds to prevent token-reconstruction attacks.
2. **Untrusted Cloud**: Responsible for storing the obfuscated index and performing similarity matching (scoring) on encoded queries via lightweight bitwise operations (intersection and Hamming distance), ensuring that the cloud can rank similar documents without ever learning the underlying plaintext data or dense embeddings.

The project demonstrates how secure indexing and searching can be performed reliably with mathematically verifiable privacy characteristics, achieving sub-linear query scaling, an 82% lower memory footprint, and 71% lower network payload compared to standard plaintext baselines.

## Architecture & Modules

The core logic is implemented inside the `pssi/` Python package:
* `pssi/client.py`: Client-side logic for document indexing, query representation encoding, token extraction, and privacy mechanisms.
* `pssi/cloud.py`: Cloud-side logic for storing sparse indices and computing robust mathematical matching scores (substring and semantic probabilities) for ranked retrieval.
* `pssi/utils.py`: Low-level cryptographic helpers, murmur hash wrappers, and utilities.

Additional project files include:
* `paper.tex`: The full ACM LaTeX source code for the publication.
* `pssi_plot_results.py`: The official script to generate the 9 publication-quality empirical results figures for the paper.

## Installation

Ensure you have Python 3.8+ installed. You can install the required dependencies in a virtual environment using `pip`:

```bash
pip install -r requirements.txt
```

*Required dependencies include `numpy`, `mmh3`, and `matplotlib`.*

## Scripts & Usage

The repository provides scripts to demonstrate the library's capabilities and reproduce the evaluation experiments in the paper:

### 1. Interactive Demonstration
Run `demo.py` to see a complete, step-by-step walkthrough of indexing a document on the client, storing it on the cloud, submitting an obfuscated query, and receiving ranking results.

```bash
python demo.py
```

### 2. Run Experiments 
Run `verify_experiments.py` to simulate the paper's experimental metrics, verifying memory reduction, network payloads, component latency breakdown, and performing a semantic ablation study.

```bash
python verify_experiments.py
```

### 3. Generate Analytical Plots
Run `pssi_plot_results.py` to generate the 9 evaluation graphs and charts demonstrating precision, recall, latency, payload, and privacy utility tradeoffs.

```bash
python pssi_plot_results.py
```
*(Graphs will be exported to the `Pics/` directory to match the LaTeX references)*

### 4. Compile the Paper
If you have a TeX distribution (e.g., MiKTeX or TeX Live) installed, you can compile the official publication:

```bash
pdflatex paper.tex
```

## Citation

If you use this codebase or find our work helpful, please refer to our research paper (citation details to follow).
