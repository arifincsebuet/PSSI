"""
Verification of the analytically-checkable claims in the paper.

This script reproduces the quantities that follow from the PSSI mechanism
itself. It deliberately does NOT reproduce the cross-system comparison table
(Elasticsearch / Lucene / ANCE / OXT latency, memory and payload), which
requires those external systems and the BEIR corpora; see README.

Checks
  1. Per-bit LDP parameter eps_0 = ln((1-eta)/eta)                 (Lemma 1)
  2. Bit-flip noise -> bit occupancy and k-hash FPR                (Sec. V-A)
  3. Sparse index size vs. the paper's 87 MB FiQA-2018 figure      (Sec. VII-C)
  4. Leakage L_A vs. eta, Algorithm 1 Sub-procedure A              (Sec. VI)
  5. Composite-weighting ablation over lambda                      (Sec. VII-D3)
"""

import json
import math
import random
import sys
import os

import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from pssi.client import PSSIClient
from pssi.cloud import PSSICloud
from pssi.attack import build_attack_dictionary, measure_leakage
from pssi.utils import bloom_false_positive_rate, composed_epsilon, epsilon0_ldp

SEED = 42


def check(label, got, expected, tol, unit=""):
    ok = abs(got - expected) <= tol
    print(f"  [{'OK ' if ok else 'DIFF'}] {label}: {got:.4g}{unit} "
          f"(paper {expected:g}{unit})")
    return ok


def main():
    print("=" * 64)
    print(" PSSI Verification -- analytically checkable paper claims")
    print("=" * 64)

    # ---- 1. LDP parameter ------------------------------------------------
    print("\n--- 1. Per-bit local DP guarantee (Lemma 1, Corollary 3) ---")
    check("eps_0 at eta=0.05", epsilon0_ldp(0.05), 2.94, 0.01)
    print("     eps_0 across the swept range:")
    for eta in (0.02, 0.05, 0.10, 0.15, 0.20, 0.30):
        print(f"       eta={eta:<5} eps_0={epsilon0_ldp(eta):.2f}")
    print(f"     conservative composed bound s*eps_0 at s=312: "
          f"{composed_epsilon(312, 0.05):.0f} (paper 918)")

    # ---- 2. Noise, occupancy, FPR ---------------------------------------
    print("\n--- 2. Bit-flip noise, occupancy and FPR (Section V-A) ---")
    client = PSSIClient(seed=SEED, embedding_model=None)
    text = ("privacy preserving semantic search over encrypted cloud document "
            "collections using bloom filters")
    clean = client.encode_bloom_filter(client.extract_ngrams(text))
    noisy = client.encode_document(text)["bf_bits"]
    p_clean, p_noisy = len(clean) / 1024, len(noisy) / 1024
    print(f"     distinct 3-grams {len(client.extract_ngrams(text))}, "
          f"clean {len(clean)} bits -> noisy {len(noisy)} bits")
    print(f"     occupancy {p_clean:.3f} -> {p_noisy:.3f}")
    # analytic prediction p' = p(1-eta) + (1-p)eta
    pred = p_clean * (1 - client.eta) + (1 - p_clean) * client.eta
    check("post-noise occupancy vs analytic", p_noisy, pred, 0.02)
    check("k-hash FPR", bloom_false_positive_rate(len(noisy)) * 100, 0.86, 0.6, "%")

    # ---- 3. Index size ---------------------------------------------------
    print("\n--- 3. Sparse index size (Section VII-C: ~87 MB on FiQA-2018) ---")
    n_docs, bf_bits, sb_bits, int_bytes = 57638, 312, 64, 4
    mb = (bf_bits + sb_bits) * int_bytes * n_docs / 1e6
    check("FiQA-2018 index size", mb, 87, 1.5, " MB")

    # ---- 4. Leakage vs eta ----------------------------------------------
    print("\n--- 4. Leakage L_A vs eta (Algorithm 1, Sub-procedure A) ---")
    random.seed(SEED)
    vocab = ["privacy", "semantic", "search", "encrypted", "cloud", "document",
             "bloom", "filter", "retrieval", "index", "noise", "hashing",
             "vector", "query", "server", "token", "corpus", "medical",
             "finance", "science", "network", "payload", "latency", "memory",
             "attack", "adversary", "threshold", "projection", "embedding",
             "similarity"]
    docs = [" ".join(random.sample(vocab, 5)) for _ in range(40)]
    truth = [d.split() for d in docs]
    adict = build_attack_dictionary(vocab)
    print(f"     |V|={len(vocab)}, {len(docs)} held-out docs, threshold=1.0")
    print(f"     {'eta':<7}{'L_A':<9}{'precision':<12}{'avg BF bits'}")
    prev = 1.1
    monotone = True
    for eta in (0.0, 0.02, 0.05, 0.10, 0.20, 0.30):
        cl = PSSIClient(eta=eta, seed=SEED, embedding_model=None)
        enc = [cl.encode_document(d) for d in docs]
        r = measure_leakage(enc, truth, adict)
        bits = sum(len(e["bf_bits"]) for e in enc) / len(enc)
        monotone &= r["leakage"] <= prev + 1e-9
        prev = r["leakage"]
        print(f"     {eta:<7}{r['leakage']:<9.3f}{r['precision']:<12.3f}{bits:6.1f}")
    print(f"  [{'OK ' if monotone else 'DIFF'}] L_A decreases monotonically in eta")
    print("     NOTE: absolute values are not the paper's 0.35 -> 0.08. Those use")
    print("     500 BEIR documents against a 50,000-token Wikipedia vocabulary;")
    print("     this is a self-contained smoke test of the same mechanism.")

    # ---- 5. Lambda ablation ---------------------------------------------
    print("\n--- 5. Composite weighting lambda (Section VII-D3) ---")
    cl = PSSIClient(eta=0.05, seed=SEED, embedding_model=None)
    cloud = PSSICloud(proj_dim=cl.proj_dim)
    for i, d in enumerate(docs[:20]):
        cloud.store_document(f"doc{i}", cl.encode_document(d))
    q = cl.encode_query(docs[0])
    print("     lambda   top-1        score   (lambda=1 substring-only, 0 semantic-only)")
    for lam in (0.0, 0.25, 0.5, 0.75, 1.0):
        top = cloud.search(q, lam=lam, top_k=1)[0]
        print(f"     {lam:<8} {top['doc_id']:<12} {top['score']:.4f}")

    print("\n" + "=" * 64)
    print(" Cross-system results (Table I, Figs. 1-5, 9) require Elasticsearch,")
    print(" Lucene/Pyserini, ANCE/FAISS, an OXT implementation and the BEIR")
    print(" corpora. They are not reproducible from this repository alone.")
    print("=" * 64)


if __name__ == "__main__":
    main()
