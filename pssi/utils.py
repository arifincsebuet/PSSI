"""
Low-level helpers for PSSI: Bloom-filter hashing, random projection, and the
bit-flip (randomized response) noise mechanism.

Paper cross-reference
---------------------
Eq. (5)  Bloom filter insertion        -> generate_bloom_hashes
Eq. (6)  Random projection sign hash   -> get_random_projection_matrix
Eq. (7)  Bit-flip noise B~[i]          -> apply_bitflip_noise
Lemma 1  eps_0 = ln((1-eta)/eta)       -> epsilon0_ldp
"""

import math

import mmh3
import numpy as np

# Paper default configuration (Section V-D)
DEFAULT_BLOOM_SIZE = 1024      # m
DEFAULT_NUM_HASHES = 4         # k
DEFAULT_NGRAM_SIZE = 3         # n
DEFAULT_PROJ_DIM = 128         # b   (semantic hash length)
DEFAULT_EMBED_DIM = 384        # p   (all-MiniLM-L6-v2 output dimensionality)
DEFAULT_ETA = 0.05             # eta (bit-flip probability)
DEFAULT_LAMBDA = 0.5           # lambda (composite weighting)


def generate_bloom_hashes(text, num_hashes=DEFAULT_NUM_HASHES,
                          max_size=DEFAULT_BLOOM_SIZE):
    """Return the set of Bloom positions for one n-gram (Eq. 5).

    Positions are returned as a set: setting the same bit twice is idempotent,
    so a hash collision simply yields fewer than `num_hashes` distinct bits.
    """
    positions = set()
    for i in range(num_hashes):
        hash_val = mmh3.hash(text, seed=i, signed=False)
        positions.add(hash_val % max_size)
    return list(positions)


def get_random_projection_matrix(input_dim=DEFAULT_EMBED_DIM,
                                 output_dim=DEFAULT_PROJ_DIM, seed=42):
    """Fixed Gaussian projection matrix W (Eq. 6).

    Entries are drawn from N(0, 1) per the Johnson-Lindenstrauss construction.
    The matrix is deterministic given `seed` and is shared by client and
    query encoder, but is never transmitted to the server.
    """
    rng = np.random.default_rng(seed)
    return rng.standard_normal((output_dim, input_dim))


def apply_bitflip_noise(active_indices, vector_size, eta=DEFAULT_ETA, rng=None):
    """Randomized response over a *dense* bit vector, returned sparsely (Eq. 7).

    Every one of the `vector_size` positions -- not merely the active ones --
    is independently flipped with probability `eta`. This matters: flipping
    only the active bits would leak the clean support, and the 0 -> 1 flips are
    what supply the cover bits that defeat token reconstruction.

    Expected active count afterwards, writing p for the clean bit occupancy:
        p' = p(1 - eta) + (1 - p)eta
    e.g. m=1024, p=0.283, eta=0.05  ->  p'=0.305, i.e. ~312 active bits.
    """
    if not 0.0 <= eta < 0.5:
        raise ValueError("eta must lie in [0, 0.5)")
    if eta == 0.0:
        return sorted(set(active_indices))

    rng = np.random.default_rng() if rng is None else rng
    dense = np.zeros(vector_size, dtype=bool)
    dense[list(active_indices)] = True
    flips = rng.random(vector_size) < eta
    noisy = np.logical_xor(dense, flips)
    return np.flatnonzero(noisy).tolist()


def epsilon0_ldp(eta=DEFAULT_ETA):
    """Tight per-bit local-DP parameter of the bit-flip mechanism (Lemma 1).

        eps_0 = ln((1 - eta) / eta)

    eta=0.05 -> ln(19) ~= 2.944. eps_0 -> 0 as eta -> 0.5 (maximum privacy,
    zero utility) and diverges as eta -> 0 (no privacy).
    """
    if not 0.0 < eta < 0.5:
        raise ValueError("eta must lie in (0, 0.5) for a finite epsilon_0")
    return math.log((1.0 - eta) / eta)


def composed_epsilon(num_bits, eta=DEFAULT_ETA):
    """Conservative fully-composed bound s * eps_0 (Theorem 2(ii)).

    This is deliberately loose; see Remark 1 in the paper. Reported alongside
    the per-bit figure rather than in place of it.
    """
    return num_bits * epsilon0_ldp(eta)


def bloom_false_positive_rate(active_bits, bloom_size=DEFAULT_BLOOM_SIZE,
                              num_hashes=DEFAULT_NUM_HASHES):
    """k-hash FPR from the measured bit occupancy: FPR = p^k.

    At the paper defaults (312 active of m=1024, k=4) this returns ~0.86%.
    """
    p = active_bits / float(bloom_size)
    return p ** num_hashes
