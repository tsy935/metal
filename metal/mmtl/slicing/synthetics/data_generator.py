import numpy as np

from metal.utils import split_data


def get_circle_mask(X, center, radius):
    h, k = center
    mask = np.sqrt((X[:, 0] - h) ** 2 + (X[:, 1] - k) ** 2) < radius
    return mask


def generate_data(N, decision_boundary_flip):
    """ Generates data in numpy form.

    Returns: (
        [uids_train, uids_val, uids_test],
        [X_train, X_val, X_test],
        [Y_train, Y_val, Y_test]
    )
    """

    uids = list(range(N))
    X = np.random.random((N, 2)) * 2 - 1
    Y = (X[:, 0] > X[:, 1] + 0.25).astype(int) + 1

    # abberation in decision boundary
    Y[decision_boundary_flip(X)] = 1

    uid_lists, Xs, Ys = split_data(uids, X, Y, splits=[0.5, 0.25, 0.25], shuffle=True)
    return uid_lists, Xs, Ys


def generate_slice_labels(X, Y, slice_funcs):
    """
    Args:
        X: [N x D] data
        Y: [N x 1] labels
        slice_funcs [dict]: mapping slice_names to slice_fn(X),
            which returns [N x 1] mask indic. whether examples are in slice

    Returns:
        slice_labels [dict]: mapping slice_names to slice_labels [N x 1],
            where 0 labels indicate abstains
    """
    slice_labels = {}
    for slice_name, slice_fn in slice_funcs.items():
        slice_mask = slice_fn(X)
        slice_label = Y.copy()
        # if not in slice, abstain with label = 0
        slice_label[np.logical_not(slice_mask)] = 0
        slice_labels[slice_name] = slice_label

    return slice_labels
