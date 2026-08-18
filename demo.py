"""
PSSI end-to-end demonstration at the paper's default configuration
(m=1024, k=4, n=3, b=128, eta=0.05, lambda=0.5; Section V-D).

Shows: client-side encoding with bit-flip noise -> sparse upload -> cloud-side
bitwise matching -> ranked results, with the server never seeing plaintext.
"""

import sys

import numpy as np

sys.path.append('.')

from pssi.client import PSSIClient
from pssi.cloud import PSSICloud
from pssi.utils import bloom_false_positive_rate


def get_demo_embedding(is_query, dim):
    """Two deliberately close vectors, standing in for a sentence encoder.

    Used only so the semantic channel is observable without downloading a
    model; install sentence-transformers for real embeddings.
    """
    base = np.zeros(dim)
    if not is_query:   # "Samsung develops smart IoT devices"
        base[:8] = [0.12, -0.80, 0.44, 0.55, -0.21, 0.91, -0.15, 0.33]
    else:              # "smart home devices"
        base[:8] = [0.10, -0.75, 0.50, 0.50, -0.20, 0.88, -0.10, 0.30]
    return base


def main():
    print("=" * 62)
    print(" PSSI Architecture Demonstration")
    print("=" * 62)

    client = PSSIClient(seed=42, embedding_model=None)
    cloud = PSSICloud(proj_dim=client.proj_dim)

    print(f"\nConfiguration (paper Section V-D):")
    print(f"  m={client.bloom_size}  k={client.num_hashes}  n={client.n_gram_sizes[0]}"
          f"  b={client.proj_dim}  eta={client.eta}  lambda=0.5")
    print(f"  per-bit LDP guarantee  eps_0 = ln((1-eta)/eta) = {client.epsilon0():.3f}")
    print(f"  real sentence encoder in use: {client.uses_real_embeddings}")

    # ---------------------------------------------------------------- index
    doc_id, doc_text = "doc123", "Samsung develops smart IoT devices"
    print(f"\n[Client] Indexing '{doc_id}': '{doc_text}'")

    ngrams_d = client.extract_ngrams(doc_text)
    clean_bf = client.encode_bloom_filter(ngrams_d)
    doc_emb = get_demo_embedding(False, client.embed_dim)
    encoded_doc = client.encode_document(doc_text, embedding=doc_emb)

    print(f"  -> {len(ngrams_d)} distinct {client.n_gram_sizes[0]}-grams")
    print(f"  -> clean Bloom support      : {len(clean_bf)} bits")
    print(f"  -> after eta={client.eta} bit-flip : {len(encoded_doc['bf_bits'])} bits "
          f"(+{len(set(encoded_doc['bf_bits'])-clean_bf)} added, "
          f"-{len(clean_bf-set(encoded_doc['bf_bits']))} removed)")
    print(f"  -> occupancy {len(encoded_doc['bf_bits'])/client.bloom_size:.3f}, "
          f"k-hash FPR {bloom_false_positive_rate(len(encoded_doc['bf_bits']))*100:.2f}%")
    print(f"     BF_d idx: {encoded_doc['bf_bits'][:6]}...")
    print(f"     SB_d idx: {encoded_doc['sb_bits'][:6]}... "
          f"({len(encoded_doc['sb_bits'])} of b={client.proj_dim})")

    cloud.store_document(doc_id, encoded_doc)
    cloud.store_document("doc_decoy",
                         client.encode_document("Apples are tasty fruits"))
    print(f"\n[Cloud] Stored 2 documents as sparse integer arrays only.")

    # ---------------------------------------------------------------- query
    query_text = "smart home devices"
    print(f"\n[Client] Query: '{query_text}'")
    q_emb = get_demo_embedding(True, client.embed_dim)
    encoded_q = client.encode_query(query_text, embedding=q_emb)
    print(f"  -> BF_q {len(encoded_q['bf_bits'])} bits, "
          f"SB_q {len(encoded_q['sb_bits'])} bits (noise applied identically)")

    print(f"\n[Cloud] Matching via set intersection + Hamming distance...")
    results = cloud.search(encoded_q, lam=0.5)

    print("\n" + "=" * 62)
    print(" Ranking (Score = lambda*P_substr + (1-lambda)*P_sem, Eq. 10)")
    print("=" * 62)
    for rank, res in enumerate(results, 1):
        print(f"Rank {rank}: {res['doc_id']}")
        print(f"   Score {res['score']:.4f} | P_substr {res['p_substr']:.4f}"
              f" | P_sem {res['p_semantic']:.4f}")

    print("\nThe cloud saw only integer arrays: no tokens, no term frequencies,")
    print("no positions, and no dense embedding values.")


if __name__ == "__main__":
    main()
