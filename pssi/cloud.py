class PSSICloud:
    def __init__(self, proj_dim=64):
        self.proj_dim = proj_dim
        self.index = {}  # In-memory "Firestore" substitute: doc_id -> (bf_bits, sb_bits)

    def store_document(self, doc_id, index_data):
        """
        Stores sparse presentation: BF_d (active bits), SB_d (active bits)
        """
        self.index[doc_id] = index_data

    def calculate_p_substr(self, bf_q, bf_d):
        set_q = set(bf_q)
        set_d = set(bf_d)
        if len(set_q) == 0:
            return 0.0
        overlap = set_q.intersection(set_d)
        return len(overlap) / float(len(set_q))

    def calculate_p_semantic(self, sb_q, sb_d):
        set_q = set(sb_q)
        set_d = set(sb_d)
        
        # In sparse form, Hamming distance computation:
        # Distance is number of bits in Q not in D + number of bits in D not in Q
        # Because we only track 1s, bits that are 0 are not in the sets
        only_in_q = len(set_q - set_d)
        only_in_d = len(set_d - set_q)
        hamming_dist = only_in_q + only_in_d
        
        # P_semantic = 1 - (H / r)
        return max(0.0, 1.0 - (hamming_dist / float(self.proj_dim)))

    def search(self, query_data, alpha=0.5, beta=0.5, top_k=5):
        bf_q = query_data["bf_bits"]
        sb_q = query_data["sb_bits"]
        
        results = []
        for doc_id, doc_data in self.index.items():
            bf_d = doc_data["bf_bits"]
            sb_d = doc_data["sb_bits"]
            
            p_substr = self.calculate_p_substr(bf_q, bf_d)
            p_semantic = self.calculate_p_semantic(sb_q, sb_d)
            
            score = alpha * p_substr + beta * p_semantic
            results.append({
                "doc_id": doc_id,
                "score": score,
                "p_substr": p_substr,
                "p_semantic": p_semantic
            })
            
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]
