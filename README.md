# Privacy-Preserving Semantic Search Index (PSSI)

This repository contains the prototype implementation of the **Privacy-Preserving Semantic Search Index (PSSI)**, which accompanies the research paper. 

PSSI is an architecture designed to enable secure and privacy-preserving substring and semantic token matching over encrypted data on an untrusted cloud server.

## Overview

The PSSI architecture strictly limits processing bounds by operating in two distinct spheres:
1. **Trusted Client**: Responsible for extracting n-grams/tokens from documents and queries, and building obfuscated sparse representations using specific hashing techniques (such as Bloom Filters).
2. **Untrusted Cloud**: Responsible for storing the obfuscated index and performing similarity matching (scoring) on encoded queries, ensuring that the cloud can rank similar documents without ever learning the underlying plaintext data.

The project demonstrates how secure indexing and searching can be performed reliably with mathematically verifiable privacy characteristics.

## Architecture & Modules

The core logic is implemented inside the `pssi/` Python package:
* `pssi/client.py`: Client-side logic for document indexing, query representation encoding, token extraction, and privacy mechanisms.
* `pssi/cloud.py`: Cloud-side logic for storing sparse indices and computing robust mathematical matching scores (substring and semantic probabilities) for ranked retrieval.
* `pssi/utils.py`: Low-level cryptographic helpers, murmur hash wrappers, and utilities.

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
Run `verify_experiments.py` to validate the analytical model's statistical mechanics, measure false positive/negative parameters, and run stress tests to ensure privacy constraints hold experimentally.

```bash
python verify_experiments.py
```

### 3. Generate Analytical Plots
Run `plot_results.py` to generate evaluation graphs and charts demonstrating precision, recall, and processing overhead.

```bash
python plot_results.py
```
*(Graphs will be exported to the `plots/` directory)*

## Citation

If you use this codebase or find our work helpful, please refer to the corresponding research paper (citation details to follow).
