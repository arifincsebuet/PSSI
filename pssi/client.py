"""
Trusted-client side of PSSI (paper Section IV-B).

Everything touching plaintext happens here: normalisation, n-gram extraction,
embedding, random projection, and bit-flip noise. Only sparse integer arrays
of *noisy* active bit indices ever leave this class.
"""

import numpy as np

from pssi.utils import (
    DEFAULT_BLOOM_SIZE, DEFAULT_EMBED_DIM, DEFAULT_ETA, DEFAULT_NGRAM_SIZE,
    DEFAULT_NUM_HASHES, DEFAULT_PROJ_DIM, apply_bitflip_noise,
    epsilon0_ldp, generate_bloom_hashes, get_random_projection_matrix,
)

# Model named in the paper's experimental setup (Section VII-A).
SENTENCE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class PSSIClient:
    """Client-side encoder.

    Defaults are the paper's configuration (Section V-D):
        m = 1024, k = 4, n = 3, b = 128, eta = 0.05.

    Parameters
    ----------
    eta : float
        Bit-flip probability. eta=0 disables noise, giving the "Plain BF"
        ablation baseline. The tight per-bit local-DP guarantee is
        eps_0 = ln((1-eta)/eta) (Lemma 1).
    embedding_model : str or None
        If sentence-transformers is installed, the named model supplies real
        semantic embeddings. Otherwise the client falls back to a deterministic
        pseudo-embedding (see `get_dummy_embedding`), which exercises the
        pipeline but carries no semantic signal.
    seed : int or None
        Seeds the noise RNG. Set for reproducible runs; leave None in
        deployment so noise is unpredictable.
    """

    def __init__(self, n_gram_sizes=(DEFAULT_NGRAM_SIZE,),
                 bloom_size=DEFAULT_BLOOM_SIZE, num_hashes=DEFAULT_NUM_HASHES,
                 embed_dim=DEFAULT_EMBED_DIM, proj_dim=DEFAULT_PROJ_DIM,
                 eta=DEFAULT_ETA, embedding_model=SENTENCE_MODEL, seed=None):
        self.n_gram_sizes = tuple(n_gram_sizes)
        self.bloom_size = bloom_size
        self.num_hashes = num_hashes
        self.embed_dim = embed_dim
        self.proj_dim = proj_dim
        self.eta = eta
        self.seed = seed
        self._rng = np.random.default_rng(seed)
        self.R = get_random_projection_matrix(self.embed_dim, self.proj_dim)

        self._encoder = None
        if embedding_model:
            try:                                   # optional dependency
                from sentence_transformers import SentenceTransformer
                self._encoder = SentenceTransformer(embedding_model)
                self.embed_dim = self._encoder.get_sentence_embedding_dimension()
                self.R = get_random_projection_matrix(self.embed_dim, self.proj_dim)
            except Exception:
                self._encoder = None               # fall back silently

    # ---------------------------------------------------------------- utils

    @property
    def uses_real_embeddings(self):
        return self._encoder is not None

    def epsilon0(self):
        """Tight per-bit LDP parameter for this client's eta (Lemma 1)."""
        return epsilon0_ldp(self.eta)

    # ------------------------------------------------------------- encoding

    def extract_ngrams(self, text):
        """Normalise and decompose into overlapping character n-grams (Eq. 4)."""
        text = "".join(ch for ch in text.lower() if ch.isalnum())
        ngrams = set()
        for n in self.n_gram_sizes:
            for i in range(len(text) - n + 1):
                ngrams.add(text[i:i + n])
        return ngrams

    def encode_bloom_filter(self, ngrams):
        """Clean (noise-free) Bloom support for a set of n-grams (Eq. 5)."""
        bf_indices = set()
        for g in ngrams:
            bf_indices.update(
                generate_bloom_hashes(g, self.num_hashes, self.bloom_size))
        return bf_indices

    def embed(self, text):
        """Dense semantic embedding; real model when available."""
        if self._encoder is not None:
            return np.asarray(self._encoder.encode(text), dtype=float)
        return self.get_dummy_embedding(text)

    def get_dummy_embedding(self, text):
        """Deterministic stand-in embedding used when no encoder is installed.

        Seeded from a stable digest -- NOT Python's built-in hash(), which is
        randomised per process unless PYTHONHASHSEED is fixed and would
        therefore make runs irreproducible.

        This carries no semantic signal: unrelated strings receive independent
        vectors. It exercises the pipeline but cannot reproduce the paper's
        retrieval-quality numbers, which require the real encoder.
        """
        import hashlib
        digest = hashlib.sha256(text.encode("utf-8")).digest()[:8]
        seed = int.from_bytes(digest, "big") % (2 ** 32)
        return np.random.default_rng(seed).standard_normal(self.embed_dim)

    def binarize_projection(self, vector):
        """Clean sign-of-projection semantic hash support (Eq. 6)."""
        z = np.dot(self.R, np.asarray(vector, dtype=float))
        return set(np.flatnonzero(z > 0).tolist())

    def encode_document(self, text, embedding=None, apply_noise=True):
        """Full client pipeline -> sparse noisy index entry I_d (Eq. 8).

        Returns active bit indices of the *noisy* representations; the clean
        supports never leave this method.
        """
        embedding = self.embed(text) if embedding is None else embedding
        bf_clean = self.encode_bloom_filter(self.extract_ngrams(text))
        sb_clean = self.binarize_projection(embedding)

        if apply_noise and self.eta > 0:
            bf_bits = apply_bitflip_noise(bf_clean, self.bloom_size,
                                          self.eta, self._rng)
            sb_bits = apply_bitflip_noise(sb_clean, self.proj_dim,
                                          self.eta, self._rng)
        else:
            bf_bits, sb_bits = sorted(bf_clean), sorted(sb_clean)

        return {"bf_bits": bf_bits, "sb_bits": sb_bits}

    def encode_query(self, text, embedding=None, apply_noise=True):
        """Queries traverse the identical pipeline (Section IV-C)."""
        return self.encode_document(text, embedding, apply_noise)
