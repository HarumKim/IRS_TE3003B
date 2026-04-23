"""
LBG (Linde-Buzo-Gray) algorithm for vector quantizer design.
Trains on LSF vectors using Euclidean distance
"""

import numpy as np


def _nearest(vectors, codebook):
    # Calculates the Euclidean distance from every data point to our current cluster centers (codebook) 
    # Returns which center each point is closest to
    diff = vectors[:, None, :] - codebook[None, :, :]
    d2 = np.einsum('nkd,nkd->nk', diff, diff)
    labels = np.argmin(d2, axis=1)
    mind2 = d2[np.arange(len(vectors)), labels]
    return labels, mind2


def lbg(vectors, codebook_size, epsilon=0.01, max_iter=50, tol=1e-4,
        rng=None, verbose=False):
    # The Linde-Buzo-Gray algorithm. It takes thousands of LSF frames and compresses them down
    # into a small dictionary of representative sounds.
    if rng is None:
        rng = np.random.default_rng(0)

    vectors = np.asarray(vectors, dtype=np.float64)
    N, d = vectors.shape

    # Start with 1 giant cluster right in the middle (average) of all our data points.
    codebook = np.mean(vectors, axis=0, keepdims=True)

    while len(codebook) < codebook_size:
        # Take our current clusters and split them in half by moving them slightly apart
        # 1 cluster becomes 2. Next loop, 2 become 4. Then 4 to 8, etc.
        perturb = epsilon * (np.abs(codebook) + 1e-6)
        codebook = np.vstack([codebook + perturb, codebook - perturb])

        # K-means
        # Now that we split them, we adjust their positions iteratively until they sit perfectly 
        # in the center of their new respective groups.
        prev_dist = np.inf
        for it in range(max_iter):
            labels, mind2 = _nearest(vectors, codebook)
            new_cb = codebook.copy()
            for k in range(len(codebook)):
                mask = labels == k
                if mask.any():
                    new_cb[k] = vectors[mask].mean(axis=0)
                else:
                    # If a cluster ended up with no points, move it to where the largest group of points is
                    counts = np.bincount(labels, minlength=len(codebook))
                    big = int(np.argmax(counts))
                    jitter = epsilon * (np.abs(codebook[big]) + 1e-6)
                    new_cb[k] = codebook[big] + jitter * rng.standard_normal(d)

            codebook = new_cb
            dist = float(np.mean(mind2))
            if verbose:
                print(f"   size={len(codebook)} iter={it} dist={dist:.5f}")
            if abs(prev_dist - dist) / (dist + 1e-12) < tol:
                break
            prev_dist = dist

    return codebook[:codebook_size]
