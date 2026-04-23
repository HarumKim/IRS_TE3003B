import os
import glob
import numpy as np

from lpc_lsf import lsf_to_lpc
from vq_lbg import lbg

FEATURES_DIR = "features"
CODEBOOKS_DIR = "codebooks"
WORDS = ["start", "stop", "left", "right", "forward",
         "back", "lift", "lower", "fast", "slow"]
CODEBOOK_SIZES = [16, 32, 64]
N_TRAIN = 10


def _idx_of(fname):
    # Helper function to extract the file number
    stem = os.path.splitext(os.path.basename(fname))[0]
    return int(stem.split("_")[-1])


def training_files(word):
    # Grabs only the first 10 files for each word to use as our training set
    all_files = sorted(glob.glob(os.path.join(FEATURES_DIR, word, f"{word}_*.npz")),
                       key=_idx_of)
    return [f for f in all_files if _idx_of(f) <= N_TRAIN]

# Create a matrix containing every sound frame recorded for a word across all 10 training files
def load_training_lsfs(word):
    lsfs = []
    for f in training_files(word):
        data = np.load(f)
        lsfs.append(data["lsfs"])
    if not lsfs:
        return np.zeros((0, 12))
    return np.vstack(lsfs) # Stack them all vertically into one massive 2D matrix

# Groups thousands of acoustic frames into a small 'codebook' of representative sounds
def train_word_codebook(word, size):
    lsfs = load_training_lsfs(word)
    if len(lsfs) < size:
        print(f"   [!] {word}: only {len(lsfs)} LSF vectors, "
              f"insufficient for size {size}")
    cb_lsf = lbg(lsfs, size) # Uses the LBG algorithm to find the best central clusters 
    cb_lsf = lbg(lsfs, size, verbose=True) # Uses the LBG algorithm to find the best central clusters 
    cb_lsf = np.sort(cb_lsf, axis=1)                 # LSF always sorted (mathematical requirement to keep the filter stable)
    cb_lpc = np.array([lsf_to_lpc(v) for v in cb_lsf]) # Convert the grouped LSFs back to LPCs for the recognition phase later
    return cb_lsf, cb_lpc


def main():
    os.makedirs(CODEBOOKS_DIR, exist_ok=True)
    for size in CODEBOOK_SIZES:
        print(f"\n=== Training codebooks size {size} ===")
        # Creates folders for each codebook size (16, 32, 64)
        out_dir = os.path.join(CODEBOOKS_DIR, f"size_{size}")
        os.makedirs(out_dir, exist_ok=True)
        for word in WORDS:
            print(f"   -> Training word '{word}'...")
            cb_lsf, cb_lpc = train_word_codebook(word, size)
            # Save our trained dictionaries compressed in a .npz file
            np.savez(os.path.join(out_dir, f"{word}.npz"),
                     lsf=cb_lsf, lpc=cb_lpc)
            print(f"   [ok] {word}: {len(cb_lpc)} codevectors")
            print(f"   [ok] {word} terminado: {len(cb_lpc)} codevectors\n")


if __name__ == "__main__":
    main()
