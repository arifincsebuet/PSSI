import mmh3
import numpy as np

def generate_bloom_hashes(text, num_hashes, max_size):
    """Generate k valid hash positions for a string."""
    positions = set()
    for i in range(num_hashes):
        # We use a stable seed equal to i
        hash_val = mmh3.hash(text, seed=i, signed=False)
        positions.add(hash_val % max_size)
    return list(positions)

def get_random_projection_matrix(input_dim, output_dim, seed=42):
    """
    Returns a deterministic random matrix with entries drawn from a standard normal distribution.
    Used for Random Projection Layer.
    """
    np.random.seed(seed)
    # Using Gaussian distribution for Johnson-Lindenstrauss
    return np.random.randn(output_dim, input_dim)
