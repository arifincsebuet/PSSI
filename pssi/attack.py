"""
Algorithm 1, Sub-procedure A: token-reconstruction attack (paper Section VI).

Threat model (Definition 1): an honest-but-curious server holding the stored
index and the Bloom hash functions {h_1..h_k}, but not the projection matrix,
the embeddings, or the plaintext. It attempts to recover document tokens.

Attack. Offline, the adversary builds A(t), the set of Bloom positions a token
t would occupy. Against a stored entry I_d it scores every vocabulary token by
containment,

    s(t) = |A(t) n I_d| / |A(t)|,

and asserts the token set S = { t : s(t) >= threshold }. With threshold = 1
this is exactly a Bloom membership query: every bit of t must be present.

Leakage L_A is the fraction of ground-truth tokens the adversary recovers
(recall). `precision` is reported alongside so the metric cannot be gamed by
asserting the whole vocabulary.

Why bit-flip noise defeats this. A token spanning j n-grams occupies up to k*j
bits, all of which must survive for membership to hold; that happens with
probability (1-eta)^(k*j), which decays fast in j. At eta=0.05 a ten-n-gram
token survives with probability ~0.95^40 ~= 0.13.
"""

from pssi.utils import (
    DEFAULT_BLOOM_SIZE, DEFAULT_NGRAM_SIZE, DEFAULT_NUM_HASHES,
    generate_bloom_hashes,
)


def _token_ngrams(token, n=DEFAULT_NGRAM_SIZE):
    token = "".join(ch for ch in token.lower() if ch.isalnum())
    if not token:
        return set()
    if len(token) < n:
        return {token}
    return {token[i:i + n] for i in range(len(token) - n + 1)}


def build_attack_dictionary(vocabulary, num_hashes=DEFAULT_NUM_HASHES,
                            bloom_size=DEFAULT_BLOOM_SIZE,
                            n=DEFAULT_NGRAM_SIZE):
    """Phase 1: A(t) for every token in the adversary's vocabulary.

    Computable offline from an auxiliary corpus, since the hash functions are
    assumed known under Definition 1.
    """
    table = {}
    for t in vocabulary:
        grams = _token_ngrams(t, n)
        positions = set()
        for g in grams:
            positions.update(generate_bloom_hashes(g, num_hashes, bloom_size))
        table[t.lower()] = frozenset(positions)
    return table


def score_tokens(bf_indices, attack_dict):
    """Containment score s(t) for every candidate token."""
    stored = set(bf_indices)
    return {t: (len(pos & stored) / float(len(pos)) if pos else 0.0)
            for t, pos in attack_dict.items()}


def recover_tokens(bf_indices, attack_dict, threshold=1.0):
    """Token set the adversary asserts is present in this document."""
    return {t for t, s in score_tokens(bf_indices, attack_dict).items()
            if s >= threshold}


def measure_leakage(encoded_docs, ground_truth_tokens, attack_dict,
                    threshold=1.0):
    """Sub-procedure A end-to-end.

    Returns
    -------
    dict with keys:
        leakage   : L_A, fraction of ground-truth tokens recovered (recall)
        precision : fraction of asserted tokens that were genuinely present
        recovered : total asserted tokens
        total     : total ground-truth tokens
    """
    hit = asserted = total = 0
    for entry, tokens in zip(encoded_docs, ground_truth_tokens):
        truth = {t.lower() for t in tokens}
        found = recover_tokens(entry["bf_bits"], attack_dict, threshold)
        hit += len(found & truth)
        asserted += len(found)
        total += len(truth)
    return {
        "leakage": hit / float(total) if total else 0.0,
        "precision": hit / float(asserted) if asserted else 0.0,
        "recovered": asserted,
        "total": total,
    }
