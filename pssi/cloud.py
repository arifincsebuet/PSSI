"""
Untrusted-cloud side of PSSI (paper Section IV-C).

The server sees only sparse integer arrays of noisy active bit indices. It
performs set intersection and Hamming distance -- integer arithmetic only --
and never holds raw tokens, term frequencies, or dense embeddings.
"""

from pssi.utils import DEFAULT_LAMBDA, DEFAULT_PROJ_DIM


class PSSICloud:
    """Sparse index store plus two-stage matching.

    Parameters
    ----------
    proj_dim : int
        Semantic hash length b; the Hamming-similarity denominator (Eq. 9).
    """

    def __init__(self, proj_dim=DEFAULT_PROJ_DIM):
        self.proj_dim = proj_dim
        self.index = {}   # doc_id -> {"bf_bits": [...], "sb_bits": [...]}

    def store_document(self, doc_id, index_data):
        """Store one sparse index entry I_d = (BF_d^idx, SB_d^idx)."""
        self.index[doc_id] = index_data

    def calculate_p_substr(self, bf_q, bf_d):
        """Bloom overlap score (Eq. 8): |BF_d n BF_q| / |BF_q|."""
        set_q, set_d = set(bf_q), set(bf_d)
        if not set_q:
            return 0.0
        return len(set_q & set_d) / float(len(set_q))

    def calculate_p_semantic(self, sb_q, sb_d):
        """Hamming similarity (Eq. 9): 1 - Ham(SB_d, SB_q) / b.

        For sparse supports the Hamming distance is the symmetric difference,
            Ham = |A| + |B| - 2|A n B|,
        which is exactly the expression given in the paper.
        """
        set_q, set_d = set(sb_q), set(sb_d)
        hamming_dist = len(set_q) + len(set_d) - 2 * len(set_q & set_d)
        return max(0.0, 1.0 - hamming_dist / float(self.proj_dim))

    def search(self, query_data, lam=DEFAULT_LAMBDA, top_k=5):
        """Rank by the composite score of Eq. (10):

            Score(d,q) = lam * P_substr + (1 - lam) * P_sem

        `lam` is a single convex weight, matching the paper. lam=1 gives the
        substring-only ablation; lam=0 gives semantic-only.
        """
        if not 0.0 <= lam <= 1.0:
            raise ValueError("lam must lie in [0, 1]")

        bf_q = query_data["bf_bits"]
        sb_q = query_data["sb_bits"]

        results = []
        for doc_id, doc_data in self.index.items():
            p_substr = self.calculate_p_substr(bf_q, doc_data["bf_bits"])
            p_semantic = self.calculate_p_semantic(sb_q, doc_data["sb_bits"])
            results.append({
                "doc_id": doc_id,
                "score": lam * p_substr + (1.0 - lam) * p_semantic,
                "p_substr": p_substr,
                "p_semantic": p_semantic,
            })

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]
