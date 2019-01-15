import os
from random import shuffle

import matplotlib.pyplot as plt
import numpy as np


def shuffle_matrices(matrices):
    """Shuffle each member of a list of matrices having the same first dimension
    (along first dimension) according to the same shuffling order.
    """

    N = matrices[0].shape[0]
    idxs = list(range(N))
    shuffle(idxs)
    out = []
    for M in matrices:
        if M.shape[0] != N:
            raise ValueError("All matrices must have same first dimension.")
        out.append(M[idxs])
    return out


def generate_multi_mode_data(n, mus, props, labels, variances, mv_normal=True):
    """Generate multi-mode data

    Args:
        - n: [int] Number of data points to generate
        - mus: [list of d-dim np.arrays] centers of the modes
        - props: [list of floats] proportion of data in each mode
        - labels: [list of ints] class label of each mode
        - variances: [list of floats] variance for each mode

    Returns:
        - X: [n x d-dim array] Data points
        - Y: [n-dim array] Data labels
        - C: [n-dim array] Index of the mode each data point belongs to
    """

    assert sum(props) == 1.0
    ns = [int(n * prop) for prop in props]
    d = mus[0].shape[0]
    I_d = np.diag(np.ones(d))

    # Generate data
    if mv_normal:
        Xu = [
            np.random.multivariate_normal(mu, I_d * var, size=ni)
            for mu, ni, var in zip(mus, ns, variances)
        ]

    else:
        # Code for computing uniformly distributed circle as modes
        Xu = []
        for mu, ni, var in zip(mus, ns, variances):
            length = np.sqrt(np.random.uniform(0, 1, ni)) * var
            angle = np.pi * np.random.uniform(
                0, 2, ni
            )  # cover full range from 0 to 2pi
            x = length * np.cos(angle) + mu[0]
            y = length * np.sin(angle) + mu[1]
            Xu.append(np.vstack((x, y)).T)

    Yu = [l * np.ones(ni) for ni, l in zip(ns, labels)]
    Cu = [i * np.ones(ni) for i, ni in enumerate(ns)]

    # Generate labels and shuffle
    return shuffle_matrices([np.vstack(Xu), np.hstack(Yu), np.hstack(Cu)])


def create_circular_slice(X, Y, C, h, k, r, slice_label, lf_num=None):
    """ Given generated data, creates a slice  (represented by 2D circle)
    by assigning all points within circle of specified location/size
    to this slice rom 1 --> 2 in-place.

    Args:
        - X: [2 x d-dim array] Data points
        - Y: [2-dim array] Data labels
        - C: [2-dim array] Index of the mode each data point belongs to
        - h: [float] horizontal shift of slice
        - k: [float] vertical shift of slice
        - r: [float] radius of slice
        - slice_label: [int] label to assign slice, in {1, -1}
    """

    circ_idx = np.sqrt((X[:, 0] - h) ** 2 + (X[:, 1] - k) ** 2) < r
    C[circ_idx] = lf_num
    Y[circ_idx] = slice_label


def lf_slice_proportion_to_radius(
    target_sp, X, C, head_config, step_size=0.05, verbose=False
):
    """ Naively estimate radius to achieve head slice / head slice + torso
     slice proportion
    """

    if target_sp == 0:
        return 0

    h = head_config["h"]
    k = head_config["k"]

    # increase radius of slice until slice proportion reaches target
    emp_sp = -1
    r = 0
    while emp_sp < target_sp:
        circ_idx = np.sqrt((X[:, 0] - h) ** 2 + (X[:, 1] - k) ** 2) < r

        # num points in S2 / num points in S1 or S2
        emp_sp = np.sum(circ_idx) / np.sum(np.logical_or(C == 1, C == 2))

        r += step_size

    if verbose:
        print(f"target sp: {target_sp}, found sp: {emp_sp}, found r: {r}")
    return r


def lf_circ_idx_for_slice_recall(
    target_val, X, slice_mask, lf_center, step_size=0.01, verbose=False
):
    """Identifies appropriate indexes to achieve target recall on
    specified slice_idx. LFs will target in shape of circle."""

    # ensure there are some elements specified in slice
    if np.sum(slice_mask) == 0:
        return np.zeros(len(X), dtype=bool)

    h, k = lf_center
    emp_recall = -1

    # shift circle to the left until precision decreases to desired amount
    r = 0
    circ_idx = np.array([])
    while emp_recall < target_val:
        circ_idx = np.sqrt((X[:, 0] - h) ** 2 + (X[:, 1] - k) ** 2) < r
        # LF recall = LF votes AND vote in slice / total in slice
        emp_recall = np.sum(np.logical_and(circ_idx, slice_mask)) / np.sum(
            slice_mask
        )

        # increase radius
        r += step_size

    if verbose:
        print(
            f"target recall: {target_val}, found recall: {emp_recall}, found r: {r}"
        )
    return circ_idx


def lf_circ_idx_for_slice_precision(
    target_val, X, slice_mask, lf_center, radius, step_size=0.01, verbose=False
):
    """Identifies appropriate indexes to achieve target precision on
    specified slice_idx. LFs will target in shape of circle."""

    assert (
        np.sum(slice_mask) > 0
    )  # ensure there are some elements specified in slice

    if target_val == 0:
        return 0

    h, k = lf_center

    emp_precision = np.inf
    r = radius
    circ_idx = np.array([])

    # shift circle to the left until precision decreases to desired amount
    while emp_precision > target_val:
        circ_idx = np.sqrt((X[:, 0] - h) ** 2 + (X[:, 1] - k) ** 2) < r
        # LF recall = LF votes AND vote in slice / total in LF2
        emp_precision = np.sum(np.logical_and(circ_idx, slice_mask)) / np.sum(
            circ_idx
        )

        # shift to left
        h -= step_size

    if verbose:
        print(
            f"target precision: {target_val}, found precision: {emp_precision}, found r: {r}"
        )
    return circ_idx


def update_L_to_target_slice(
    L,
    X,
    C,
    target_metric,
    target_val,
    lf_num,
    acc,
    center,
    slice_radius,
    slice_label,
):
    """ Updates L matrix in place to target slice with specified precision/lf score.

    Args:
        L: [N x num_lfs] label matrix that will be updated in place
        X: [N x d] all data points
        C: [N x 1] slice num for each data point
        target_metric: [string] either precision or recall of LF over slice
        target_val: [float] precision or recall value that you want to target
        lf_num: [int] index i for \lamdba_i targeting \slice_i
        acc: [float] accuracy of this LF
        center: [tuple of floats]] (x, y) coordinates slice
        slice_radius: [float] default size of slice radius
        slice_label: [int] either +1/-1 for label to assign the slice
    """
    assert target_metric in ["precision", "recall"]

    N = X.shape[0]

    # set LF values based on circle that satisfies target metric (ex 0.7 precision)
    if target_metric == "precision":
        lf_idx = lf_circ_idx_for_slice_precision(
            target_val, X, C == lf_num, center, radius=slice_radius
        )
    elif target_metric == "recall":
        lf_idx = lf_circ_idx_for_slice_recall(
            target_val, X, C == lf_num, center
        )

    L[lf_idx, lf_num] = slice_label

    # set some labels incorrectly (-slice_label) based on accuracy
    wrong_mask = np.random.random(N) > acc
    L[np.logical_and(wrong_mask, lf_idx), lf_num] = -slice_label


def generate_perfect_cov_L(N, accs, Y, C):
    """Generate label matrix. We assume that the last LF is the head LF and the
    one before it is the torso LF it will interact with.

    Args:
        - N: [int] Number of data points
        - accs: [list of floats] accuracies of LFs
        - Y: [n-dim array] Data labels
        - C: [n-dim array] Index of the mode each data point belongs to

    Returns:
        - L: [n x d-dim array] Data points
    """
    m = np.shape(accs)[0]

    # Construct a label matrix with given accs and covs
    L = np.zeros((N, m))
    for i in range(N):
        j = int(C[i])  # slice
        if np.random.random() < accs[j]:
            L[i, j] = Y[i]
        else:
            L[i, j] = -Y[i]

    return L


def generate_imperfect_L(
    N, X, C, accs, mus, labels, lf_metrics, head_config=None
):
    """ Generates imperfect L matrix with specified precision or recall
    over the slice of interest.

    Args:
        N: num data points
        X: [N, d] train data
        C: [N, 1] identifies slice assignment of each datapoint
        mus: [num_slices, 2] identifies slice centers
        lf_metrics: [list of tuples] for each lf, specifies (metric, value)
        head_config: dict of head config values for slice of interest

    Returns:
        L matrix [N, num_lfs]
    """
    m = len(accs)

    # init L matrix
    L = np.zeros((N, m))

    # update L matrix based on config settings for "normal" (non circular slice) LFs
    num_normal_lfs = len(labels)
    for lf_num in range(num_normal_lfs):
        target_metric, target_val = lf_metrics[lf_num]
        update_L_to_target_slice(
            L,
            X,
            C,
            target_metric,
            target_val,
            lf_num=lf_num,
            acc=accs[lf_num],
            center=tuple(mus[lf_num]),
            slice_radius=None,
            slice_label=labels[lf_num],
        )

    # update L matrix for circular slice "head"
    if head_config:
        lf_num = m - 1
        target_metric, target_val = lf_metrics[lf_num]
        update_L_to_target_slice(
            L,
            X,
            C,
            target_metric,
            target_val,
            lf_num=lf_num,
            acc=accs[lf_num],
            center=(head_config["h"], head_config["k"]),
            slice_radius=head_config["r"],
            slice_label=head_config["slice_label"],
        )

    return L


def generate_synthetic_data(config, x_var=None, x_val=None, verbose=False):
    """ Generates synthetic data, overwriting default "x_var"
    in config with "x_val" if they are specified.

    Args:
        config: with default data generation values
        x_var: variable to override, in {"sp", "acc", "cov"}
        x_val value to override variable with

    Returns:
        X: data points in R^2
        Y: labels in {-1, 1}
        C: slice assignment in {0, 1, 2}
        L: generated label matrix (n x 2)
    """
    assert x_var in ["sp", "acc", "cov.precision", "cov.recall", None]

    X, Y, C = generate_multi_mode_data(
        config["N"],
        config["mus"],
        config["props"],
        config["labels"],
        config["variances"],
    )

    if config["head_config"]:
        # overwrite data points to create head slice
        # find radius for specified overlap proportion
        slice_radius = (
            lf_slice_proportion_to_radius(
                x_val, X, C, config["head_config"], verbose=verbose
            )
            if x_var == "sp"
            else config["head_config"]["r"]
        )

        create_circular_slice(
            X,
            Y,
            C,
            h=config["head_config"]["h"],
            k=config["head_config"]["k"],
            r=slice_radius,
            slice_label=config["head_config"]["slice_label"],
            lf_num=2,
        )

    # labeling function generation
    accs = config["accs"]
    if x_var == "acc":
        accs[-1] = x_val  # vary head lf (last index) accuracy

    covs = config["covs"]
    if x_var and x_var.startswith("cov"):
        x_var, metric = tuple(x_var.split("."))  # splits to (cov, precision)
        assert metric in ["precision", "recall"]
        covs[-1] = (metric, x_val)

    if config.get("perfect_cov", False):
        L = generate_perfect_cov_L(config["N"], accs, Y, C)
    else:
        L = generate_imperfect_L(
            config["N"],
            X,
            C,
            accs,
            config["mus"],
            config["labels"],
            config["covs"],
            config["head_config"],
        )

    return X, Y, C, L
