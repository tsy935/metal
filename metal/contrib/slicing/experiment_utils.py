import numpy as np
from scipy.special import expit
from termcolor import colored

from torch.utils.data.sampler import WeightedRandomSampler
from metal.metrics import metric_score, accuracy_score

def get_weighted_sampler_via_targeting_lfs(L_train, targeting_lfs_idx, upweight_multiplier):
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
        
    upweighting_mask = np.sum(L_train[:, targeting_lfs_idx], axis=1) > 0
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


def generate_weak_labels(L_train, accs=None, verbose=False, seed=0):
    """ Combines L_train into weak labels either using accuracies of LFs or LabelModel.""" 
    L_train_np = L_train.copy()

    if accs is not None:
        accs = np.array(accs)
        if np.any(accs==1):
            print ("Warning: clipping accuracy at 0.95")
            accs[accs==1] = 0.95

        # Combine with weights computed from LF accuracies
        w = np.log(accs / (1 - accs))
        w[np.abs(w) == np.inf] = 0  # set weights from acc==0 to 0

        # L_train_pt = torch.from_numpy(L_train.astype(np.float32))
        # TODO: add multiclass support
        L_train_np[L_train_np == 2] = -1
        label_probs = expit(2 * L_train_np @ w).reshape(-1, 1)
        Y_weak = np.concatenate((label_probs, 1 - label_probs), axis=1)
    else:
        if verbose:
            print("Training MeTaL label model...")
        from metal.label_model import LabelModel

        label_model = LabelModel(k=2, seed=seed)
        L_train_np[L_train_np == -1] = 2
        label_model.train_model(L_train_np, n_epochs=500, print_every=25, verbose=verbose)
        Y_weak = label_model.predict_proba(L_train)

    return Y_weak


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
