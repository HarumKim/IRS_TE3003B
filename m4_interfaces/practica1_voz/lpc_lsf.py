"""
LPC and LSF for the voice recognition practice.
"""

import numpy as np
from scipy.linalg import toeplitz

LPC_ORDER = 12

# Autocorrelation
def autocorrelation(x, order):
    # Measures how similar a signal is to a delayed version of itself.
    # This helps find repeating patterns in the audio, like the vibration of vocal cords.
    x = np.asarray(x, dtype=np.float64)
    r = np.empty(order + 1)
    for k in range(order + 1):
        r[k] = np.dot(x[:len(x) - k], x[k:])
    return r

# Levinson-Durbin 
def levinson_durbin(r, order):
    # A highly efficient math algorithm to solve equations for our LPC coefficients.
    # 'a' represents the filter coefficients (which mathematically model the shape of the vocal tract).
    # 'err' is the prediction error
    a = np.zeros(order + 1)
    a[0] = 1.0
    err = r[0]
    if err <= 0:
        return a, 1e-12

    for i in range(1, order + 1):
        k = -(r[i] + np.dot(a[1:i], r[i - 1:0:-1])) / err
        a_new = a.copy()
        for j in range(1, i):
            a_new[j] = a[j] + k * a[i - j]
        a_new[i] = k
        a = a_new
        err = err * (1.0 - k * k)
        if err <= 0:
            err = 1e-12
            break
    return a, err


def compute_lpc(frame, order=LPC_ORDER):
    r = autocorrelation(frame, order)
    a, err = levinson_durbin(r, order)
    return a, err, r


# LPC -> LSF 
def lpc_to_lsf(a):
    # Converts LPC coefficients into Line Spectral Frequencies (LSF).
    # LPC values are very sensitive. A tiny change can make the mathematical filter blow up
    # LSF is a different representation of the exact same data, but it is much more stable, making it 
    # perfect for calculating Euclidean distances.
    a = np.asarray(a, dtype=np.float64)
    p = len(a) - 1
    a_rev = a[::-1]
    P = np.concatenate([a, [0.0]]) + np.concatenate([[0.0], a_rev])
    Q = np.concatenate([a, [0.0]]) - np.concatenate([[0.0], a_rev])

    def _angles(poly):
        roots = np.roots(poly)
        ang = np.angle(roots)
        ang = ang[(ang > 1e-6) & (ang < np.pi - 1e-6)]
        return np.sort(ang)

    lsf_P = _angles(P)
    lsf_Q = _angles(Q)
    lsf = np.sort(np.concatenate([lsf_P, lsf_Q]))

    if len(lsf) != p:
        raise ValueError(f"Expected {p} LSF, got {len(lsf)}")
    return lsf


# LSF -> LPC
def lsf_to_lpc(lsf):
    # Converts our stable LSF values back into LPC coefficients
    # We have to do this after training our codebooks, because the recognition phase requires the data to be in the LPC format to work
    lsf = np.sort(np.asarray(lsf, dtype=np.float64))
    p = len(lsf)
    if p % 2 != 0:
        raise NotImplementedError("Only even order supported (p=12)")

    lsf_P = lsf[0::2]   # evens -> P polynomial
    lsf_Q = lsf[1::2]   # odds -> Q polynomial

    P = np.array([1.0, 1.0])
    for w in lsf_P:
        P = np.convolve(P, [1.0, -2.0 * np.cos(w), 1.0])

    Q = np.array([1.0, -1.0])
    for w in lsf_Q:
        Q = np.convolve(Q, [1.0, -2.0 * np.cos(w), 1.0])

    A_full = 0.5 * (P + Q)       
    a = A_full[:p + 1]
    a[0] = 1.0                 
    return a

# Itakura-Saito 
def itakura_saito(a_ref, e_test, r_test):
    # A specialized distance measurement specifically designed for human speech.
    # Instead of measuring geometric distance, it calculates how well our 
    # trained vocal tract model matches the actual sound of the test audio frame
    # A lower score means a better match
    p = len(a_ref) - 1
    R = toeplitz(r_test[:p + 1])
    e_ref = float(a_ref @ R @ a_ref)
    if e_test <= 0 or e_ref <= 0:
        return np.inf
    ratio = e_ref / e_test
    return ratio - np.log(ratio) - 1.0
