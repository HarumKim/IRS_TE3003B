"""
Audio utilities for the speech recognition practice.
"""

import numpy as np
from scipy.io import wavfile

# Constants
FS = 16000
FRAME_LEN = 320        # 20 ms  @ 16 kHz
FRAME_HOP = 128        # 8 ms   @ 16 kHz
PREEMPH_ALPHA = 0.95

# Step 1 - Load audios
def load_wav(path):
    # Load the audio file
    fs, x = wavfile.read(path)
    if x.ndim > 1:                     # Force mono
        # If the audio is stereo (2 channels), just grab the first channel to make it mono
        x = x[:, 0]
    if np.issubdtype(x.dtype, np.integer):
        # Normalize the audio values to be floating point numbers between -1.0 and 1.0
        x = x.astype(np.float32) / np.iinfo(x.dtype).max
    else:
        x = x.astype(np.float32)
    return x, fs


# Step 2 - Pre-emphasis
def preemphasis(x, alpha=PREEMPH_ALPHA):
    # Applies a high-pass filter to the audio. 
    # Human voices have a lot of energy in low frequencies, so this boosts the high frequencies to balance the spectrum out.
    y = np.empty_like(x)
    y[0] = x[0]
    y[1:] = x[1:] - alpha * x[:-1]
    return y


# Step 3 - Windowing
def frame_signal(x, frame_len=FRAME_LEN, hop=FRAME_HOP):
    # Chops the continuous audio signal into tiny overlapping chunks of 20ms.
    if len(x) < frame_len:
        return np.zeros((0, frame_len), dtype=np.float32)
    n_frames = 1 + (len(x) - frame_len) // hop
    # A Hamming window smooths the edges of each chunk so they taper off to zero, preventing mathematical "clicks" or noise.
    window = np.hamming(frame_len).astype(np.float32)
    frames = np.empty((n_frames, frame_len), dtype=np.float32)
    for i in range(n_frames):
        s = i * hop
        frames[i] = x[s:s + frame_len] * window
    return frames


# Step 4 - Endpoing Detection
def short_time_energy(frames):
    # Calculates the energy  of each frame
    return np.sum(frames.astype(np.float64) ** 2, axis=1)


def zero_crossing_rate(frames):
    # Counts how many times the signal crosses the zero-axis. 
    # High ZCR usually means noisy, unvoiced sounds like "s" or "f".
    signs = np.sign(frames)
    signs[signs == 0] = 1
    return np.sum(np.abs(np.diff(signs, axis=1)), axis=1) / (2.0 * frames.shape[1])


def endpoint_detection(frames, n_noise_frames=8, min_voiced=3):
    # Analyzes the frames to figure out exactly where the user started and stopped talking,
    # so we can ignore the background silence at the beginning and end of the recording.
    if len(frames) == 0:
        return 0, 0

    E = short_time_energy(frames)
    Z = zero_crossing_rate(frames)

    # Noise statistics with the first frames
    n_noise = min(n_noise_frames, max(1, len(frames) // 5))
    e_noise = np.mean(E[:n_noise]) + 1e-10
    z_noise = np.mean(Z[:n_noise])

    # Thresholds: two levels for hysteresis
    e_max = np.max(E)
    ITU = max(10.0 * e_noise, 0.10 * e_max)          # High threshold
    ITL = max(4.0 * e_noise, 0.03 * e_max)           # Low threshold
    IZCT = z_noise + 0.25 * np.std(Z[:n_noise] + 1e-6)

    # Find first frame crossing ITU
    above = E > ITU
    if not above.any():
        return 0, len(frames)

    # Consecutive runs above ITU of at least min_voiced frames
    start = None
    run = 0
    for i, a in enumerate(above):
        run = run + 1 if a else 0
        if run >= min_voiced:
            start = i - min_voiced + 1
            break
    if start is None:
        start = int(np.argmax(above))

    # Step back while E > ITL
    while start > 0 and E[start - 1] > ITL:
        start -= 1
    # Step back a bit more if ZCR is still high (fricatives /s/, /f/)
    back_limit = max(0, start - 25)
    while start > back_limit and Z[start - 1] > IZCT:
        start -= 1

    # End: last frame above ITU, then extend with ITL and ZCR
    end = len(frames) - 1 - int(np.argmax(above[::-1]))
    while end < len(frames) - 1 and E[end + 1] > ITL:
        end += 1
    fwd_limit = min(len(frames) - 1, end + 25)
    while end < fwd_limit and Z[end + 1] > IZCT:
        end += 1

    return int(start), int(end + 1)
