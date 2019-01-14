import numpy as np
from scipy.special import expit
from termcolor import colored

from metal.metrics import metric_score


def generate_weak_labels(L_train, accs=None, verbose=False, seed=0):
    """ Combines L_train into weak labels either using accuracies of LFs or LabelModel.""" 
    L_train_np = L_train.copy()

    if accs is not None:
        # Combine with weights computed from LF accuracies
        w = np.log(accs / (1 - accs))
        w[np.abs(w) == np.inf] = 0  # set weights from acc==0 to 0

        # L_train_pt = torch.from_numpy(L_train.astype(np.float32))
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
