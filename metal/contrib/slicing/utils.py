from collections import Counter

import numpy as np
import torch
from scipy.sparse import csr_matrix
from termcolor import colored
from torch.utils.data.sampler import WeightedRandomSampler
from tqdm import tqdm

from metal.metrics import accuracy_score, metric_score


def add_pepper(L, pepper_pct, verbose=True):
    """Randomly flips a given percent of abstain votes on all unipolar LFs"""

    def flip(x):
        return 1 if x == 2 else 1

    k = 2
    n, m = L.shape
    L = L.copy()

    peppered = 0
    for j in range(m):
        label_vec = np.asarray(L[:, j].todense())
        label_counts = Counter(int(label) for label in label_vec)
        if len(label_counts) < k + 1:
            peppered += 1
            polarity = max(label_counts.keys())
            idxs, vals = np.where(label_vec != polarity)
            pepper_count = int(pepper_pct * label_counts[polarity])
            selected = np.random.choice(idxs, pepper_count)
            L[selected, j] = flip(polarity)
    if verbose:
        print(
            f"Added pepper={pepper_pct} random negatives on {peppered}/{m} LFs"
        )
    return L


def get_L_weights_from_targeting_lfs_idx(m, targeting_lfs_idx, multiplier):
    L_weights = np.ones(m)
    L_weights[targeting_lfs_idx] = multiplier
    L_weights = list(L_weights)
    return L_weights


def slice_mask_from_targeting_lfs_idx(L, targeting_lfs_idx):
    if isinstance(L, csr_matrix):
        L = np.array(L.todense())

    mask = np.sum(L[:, targeting_lfs_idx], axis=1) > 0
    return mask.squeeze()


def get_weighted_sampler_via_targeting_lfs(
    L_train, targeting_lfs_idx, upweight_multiplier
):
    """ Creates a weighted sampler that upweights values based on whether they are targeted
    by LFs. Intuitively, upweights examples that might be "contributing" to slice performance,
    as defined by label matrix.

    Args:
        L_train: label matrix
        targeting_lfs_idx: list of ints pointing to the columns of the L_matrix
            that are targeting the slice of interest.
        upweight_multiplier: multiplier to upweight samples covered by targeting_lfs_idx
    Returns:
        WeightedSampler to be pasesd into Dataloader

    """

    upweighting_mask = slice_mask_from_targeting_lfs_idx(
        L_train, targeting_lfs_idx
    )
    weights = np.ones(upweighting_mask.shape)
    weights[upweighting_mask] = upweight_multiplier
    num_samples = int(sum(weights))
    return WeightedRandomSampler(weights, num_samples)


def compute_lf_accuracies(L_dev, Y_dev):
    """ Returns len m list of accuracies corresponding to each lf"""
    accs = []
    m = L_dev.shape[1]
    for lf_idx in range(m):
        voted_idx = L_dev[:, lf_idx] != 0
        accs.append(accuracy_score(L_dev[voted_idx, lf_idx], Y_dev[voted_idx]))
    return accs


def generate_weak_labels(L_train, weights=None, verbose=False, seed=0):
    """ Combines L_train into weak labels either using accuracies of LFs or LabelModel."""
    raise Exception("Use the classes `metal.label_model.baselines`!")


def compare_LF_slices(
    Yp_ours, Yp_base, Y, L_test, LFs, metric="accuracy", delta_threshold=0
):
    """Compares improvements between `ours` over `base` predictions."""

    improved = 0
    for LF_num, LF in enumerate(LFs):
        LF_covered_idx = np.where(L_test[:, LF_num] != 0)[0]
        ours_score = metric_score(
            Y[LF_covered_idx], Yp_ours[LF_covered_idx], metric
        )
        base_score = metric_score(
            Y[LF_covered_idx], Yp_base[LF_covered_idx], metric
        )

        delta = ours_score - base_score
        # filter out trivial differences
        if abs(delta) < delta_threshold:
            continue

        to_print = (
            f"[{LF.__name__}] delta: {delta:.4f}, "
            f"OURS: {ours_score:.4f}, BASE: {base_score:.4f}"
        )

        if ours_score > base_score:
            improved += 1
            print(colored(to_print, "green"))
        elif ours_score < base_score:
            print(colored(to_print, "red"))
        else:
            print(to_print)

    print(f"improved {improved}/{len(LFs)}")
