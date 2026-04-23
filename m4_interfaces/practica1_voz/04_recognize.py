import os
import glob
import numpy as np
from scipy.linalg import toeplitz

FEATURES_DIR = "features"
CODEBOOKS_DIR = "codebooks"
WORDS = ["start", "stop", "left", "right", "forward",
         "back", "lift", "lower", "fast", "slow"]
CODEBOOK_SIZES = [16, 32, 64]
N_TRAIN = 10   


def _idx_of(fname):
    return int(os.path.splitext(os.path.basename(fname))[0].split("_")[-1])

def test_files(word):
    all_files = sorted(glob.glob(os.path.join(FEATURES_DIR, word, f"{word}_*.npz")),
                       key=_idx_of)
    return [f for f in all_files if _idx_of(f) > N_TRAIN]


def frame_distances_to_codebook(test_file, cb_lpc):
    # For each frame of the test file, minimum Itakura-Saito distance to any codevector of the codebook.
    # returns the mean over frames.
    data = np.load(test_file)
    test_errors = data["errors"]
    test_autos = data["autocorrs"]
    n_frames = len(test_errors)
    p = cb_lpc.shape[1] - 1

    dist_sum = 0.0
    n_valid = 0
    for e_t, r in zip(test_errors, test_autos):
        R = toeplitz(r[:p + 1])
        e_refs = np.einsum('ki,ij,kj->k', cb_lpc, R, cb_lpc)
        e_refs = np.where(e_refs > 0, e_refs, np.inf)
        if e_t <= 0:
            continue
        ratio = e_refs / e_t
        d = ratio - np.log(ratio) - 1.0
        dist_sum += float(np.min(d))
        n_valid += 1
    if n_valid == 0:
        return np.inf
    return dist_sum / n_valid


def classify(test_file, codebooks):
    # Returns the word with the lowest average distance
    best_word, best_d = None, np.inf
    for word, cb in codebooks.items():
        d = frame_distances_to_codebook(test_file, cb)
        if d < best_d:
            best_d, best_word = d, word
    return best_word


# Print confusion matrixes
def print_confusion(cm, labels):
    w = max(8, max(len(l) for l in labels) + 1)
    header = " " * w + "".join(f"{l[:w-1]:>{w}}" for l in labels)
    print(header)
    for i, lab in enumerate(labels):
        row = f"{lab:<{w}}" + "".join(f"{cm[i, j]:>{w}d}" for j in range(len(labels)))
        print(row)


def evaluate_size(size):
    print(f"\n========== Codebook size = {size} ==========")
    # load codebooks of this size
    codebooks = {}
    for word in WORDS:
        data = np.load(os.path.join(CODEBOOKS_DIR, f"size_{size}", f"{word}.npz"))
        codebooks[word] = data["lpc"]

    cm = np.zeros((len(WORDS), len(WORDS)), dtype=int)
    correct = 0
    total = 0

    for i, true_word in enumerate(WORDS):
        for tf in test_files(true_word):
            pred = classify(tf, codebooks)
            j = WORDS.index(pred)
            cm[i, j] += 1
            total += 1
            correct += int(pred == true_word)

    acc = correct / total if total else 0.0
    print(f"Accuracy: {correct}/{total} = {acc:.2%}\n")
    print("Confusion matrix (rows = true, columns = predicted):")
    print_confusion(cm, WORDS)
    return cm, acc


def main():
    results = {}
    for size in CODEBOOK_SIZES:
        cm, acc = evaluate_size(size)
        results[size] = (cm, acc)

    print("\n========== Summary ==========")
    for size, (_, acc) in results.items():
        print(f"   size {size:>3}: {acc:.2%}")


if __name__ == "__main__":
    main()
