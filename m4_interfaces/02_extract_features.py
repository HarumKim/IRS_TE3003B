import os
import glob
import numpy as np

from audio_utils import load_wav, preemphasis, frame_signal, endpoint_detection
from lpc_lsf import compute_lpc, lpc_to_lsf, LPC_ORDER

DATASET_DIR = "dataset_voz"
FEATURES_DIR = "features"
WORDS = ["start", "stop", "left", "right", "forward",
         "back", "lift", "lower", "fast", "slow"]


def process_file(wav_path):
    x, fs = load_wav(wav_path)
    assert fs == 16000, f"Expected 16 kHz, got {fs}"

    # Step 2 - Preemphasis Filter
    y = preemphasis(x) # Amplifies high frequencies to balance the sound spectrum

    # Step 3 - Hamming Window
    frames = frame_signal(y) # Slices into small chunks and smooths the edges to prevent mathematical noise

    # Step 4 - Find beginning and end of each word
    start, end = endpoint_detection(frames) # Analyses the words energy to figure out where the word starts and ends
    voiced = frames[start:end]

    # LPC, LSF, autocorrelation per frame
    lpcs, lsfs, errs, rs = [], [], [], []
    for fr in voiced:
        # Ignore almost silent frames
        if np.sum(fr ** 2) < 1e-8:
            continue
        try:
            a, e, r = compute_lpc(fr, LPC_ORDER)
            lsf = lpc_to_lsf(a)
        except Exception:
            continue
        lpcs.append(a)     # LPC represents how the human vocal tract produced that specific sound
        lsfs.append(lsf)   # LSF is a LPC transformation that's more stable and useful for grouping sounds
        errs.append(e)
        rs.append(r)

    return {
        "lpcs": np.asarray(lpcs),
        "lsfs": np.asarray(lsfs),
        "errors": np.asarray(errs),
        "autocorrs": np.asarray(rs),
        "start_frame": start,
        "end_frame": end,
    }


def main():
    os.makedirs(FEATURES_DIR, exist_ok=True)
    for word in WORDS:
        in_dir = os.path.join(DATASET_DIR, word)
        out_dir = os.path.join(FEATURES_DIR, word)
        os.makedirs(out_dir, exist_ok=True)

        files = sorted(glob.glob(os.path.join(in_dir, "*.wav")))
        if not files:
            print(f"[!] {word}: no WAV files in {in_dir}")
            continue

        print(f"\n>>> {word}  ({len(files)} files)")
        for wav in files:
            name = os.path.splitext(os.path.basename(wav))[0]
            feats = process_file(wav)
            if len(feats["lsfs"]) == 0:
                print(f"   [X] {name}: no valid frames")
                continue
            out_path = os.path.join(out_dir, f"{name}.npz")
            np.savez(out_path, **feats)
            print(f"   [ok] {name}: {len(feats['lsfs'])} frames "
                  f"(voice: {feats['start_frame']} -> {feats['end_frame']})")


if __name__ == "__main__":
    main()
