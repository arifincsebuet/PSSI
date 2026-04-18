import numpy as np
from pssi.utils import generate_bloom_hashes, get_random_projection_matrix

class PSSIClient:
    def __init__(self, n_gram_sizes=(3,), bloom_size=1024, num_hashes=3, embed_dim=128, proj_dim=64):
        self.n_gram_sizes = n_gram_sizes
        self.bloom_size = bloom_size
        self.num_hashes = num_hashes
        self.embed_dim = embed_dim
        self.proj_dim = proj_dim
        self.R = get_random_projection_matrix(self.embed_dim, self.proj_dim)

    def extract_ngrams(self, text):
        text = text.lower().replace(" ", "")
        ngrams = []
        for n in self.n_gram_sizes:
            for i in range(len(text) - n + 1):
                ngrams.append(text[i:i+n])
        return set(ngrams)

    def encode_bloom_filter(self, ngrams):
        bf_indices = set()
        for g in ngrams:
            hashes = generate_bloom_hashes(g, self.num_hashes, self.bloom_size)
            bf_indices.update(hashes)
        return list(bf_indices)

    def get_dummy_embedding(self, text):
        """
        Generates a dummy dense vector for demonstration based on a stable hash.
        """
        np.random.seed(abs(hash(text)) % (2**32))
        return np.random.randn(self.embed_dim)

    def binarize_projection(self, vector):
        z = np.dot(self.R, vector)
        sb = (z > 0).astype(int).tolist()
        sb_sparse = [i for i, val in enumerate(sb) if val == 1]
        return sb_sparse

    def encode_document(self, text, embedding=None):
        if embedding is None:
            embedding = self.get_dummy_embedding(text)
        ngrams = self.extract_ngrams(text)
        bf_sparse = self.encode_bloom_filter(ngrams)
        sb_sparse = self.binarize_projection(embedding)
        return {"bf_bits": bf_sparse, "sb_bits": sb_sparse}

    def encode_query(self, text, embedding=None):
        return self.encode_document(text, embedding)
