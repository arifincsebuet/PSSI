import sys
import numpy as np

# ensure pssi is in path
sys.path.append('.')

from pssi.client import PSSIClient
from pssi.cloud import PSSICloud

def get_demo_embedding(is_query):
    # To demonstrate high semantic similarity without an LLM, we manually construct
    # embeddings that are close in cosine distance.
    if not is_query:
        # Document "Samsung develops smart IoT devices"
        return np.array([0.12, -0.8, 0.44, 0.55, -0.21, 0.91, -0.15, 0.33] + [0.0]*120)
    else:
        # Query "smart home devices"
        return np.array([0.10, -0.75, 0.50, 0.50, -0.20, 0.88, -0.10, 0.30] + [0.0]*120)

def main():
    print("==================================================")
    print(" PSSI Architecture Demonstration")
    print("==================================================\n")

    # Initialize client and cloud
    client = PSSIClient(n_gram_sizes=[3], bloom_size=128, num_hashes=3, embed_dim=128, proj_dim=64)
    cloud = PSSICloud(proj_dim=64)

    # 1. Document Indexing (Client Side)
    doc_id = "doc123"
    doc_text = "Samsung develops smart IoT devices"
    print(f"[Client] Indexing Document '{doc_id}': '{doc_text}'")
    
    # Extract n-grams explicitly to show the process
    ngrams_d = client.extract_ngrams(doc_text)
    print(f"  -> Extracted {len(ngrams_d)} 3-grams: {list(ngrams_d)[:5]}...")
    
    doc_embedding = get_demo_embedding(is_query=False)
    encoded_doc = client.encode_document(doc_text, embedding=doc_embedding)
    print(f"  -> Obfuscated representation created!")
    print(f"     Sparse BF_d: {encoded_doc['bf_bits'][:5]}... (total {len(encoded_doc['bf_bits'])})")
    print(f"     Sparse SB_d: {encoded_doc['sb_bits'][:5]}... (total {len(encoded_doc['sb_bits'])})\n")

    # Upload to Cloud
    print(f"[Cloud] Storing representation for '{doc_id}' in sparse format...\n")
    cloud.store_document(doc_id, encoded_doc)

    # Add a decoy document
    cloud.store_document("doc_decoy", client.encode_document("Apples are tasty fruits"))

    # 2. Query Encoding (Client Side)
    query_text = "smart home devices"
    print(f"[Client] Processing Query: '{query_text}'")
    ngrams_q = client.extract_ngrams(query_text)
    print(f"  -> Extracted {len(ngrams_q)} 3-grams: {list(ngrams_q)[:5]}...")

    q_embedding = get_demo_embedding(is_query=True)
    encoded_q = client.encode_query(query_text, embedding=q_embedding)
    print(f"  -> Query representation encoded.")
    print(f"     Sparse BF_q: {encoded_q['bf_bits'][:5]}... (total {len(encoded_q['bf_bits'])})")
    print(f"     Sparse SB_q: {encoded_q['sb_bits'][:5]}... (total {len(encoded_q['sb_bits'])})\n")

    # 3. Cloud Matching & Ranking (Cloud Side)
    print(f"[Cloud] Received encoded query. Computing matching over index...")
    results = cloud.search(encoded_q, alpha=0.5, beta=0.5)

    print("\n==================================================")
    print(" Final Ranking Results")
    print("==================================================")
    for rank, res in enumerate(results, 1):
        print(f"Rank {rank}: {res['doc_id']}")
        print(f"  -> Score        : {res['score']:.4f}")
        print(f"  -> P_substr     : {res['p_substr']:.4f}")
        print(f"  -> P_semantic   : {res['p_semantic']:.4f}\n")

if __name__ == "__main__":
    main()
