import numpy as np
import torch
from tqdm import tqdm


def evaluate_slicing(
    model,
    eval_loader,
    metrics=["accuracy", "precision", "recall", "f1"],
    verbose=True,
):
    """
    Args:
        model: a trained EndModel (or subclass)
        eval_loader: a loader containing X, Y, Z
    """
    X, Y, Z = separate_eval_loader(eval_loader)
    out_dict = {}

    # Evaluating on full dataset
    if verbose:
        print(f"All: {len(Z)} examples")
    metrics_full = model.score((X, Y), metrics, verbose=verbose)
    out_dict["all"] = {metrics[i]: metrics_full[i] for i in range(len(metrics))}

    # Evaluating on slice
    slices = sorted(set(Z))
    for s in slices:

        # Getting indices of points in slice
        inds = [i for i, e in enumerate(Z) if e == s]
        if verbose:
            print(f"\nSlice {s}: {len(inds)} examples")
        X_slice = X[inds]
        Y_slice = Y[inds]

        metrics_slice = model.score(
            (X_slice, Y_slice), metrics, verbose=verbose
        )

        out_dict[f"slice_{s}"] = {
            metrics[i]: metrics_slice[i] for i in range(len(metrics_slice))
        }

    print("\nSUMMARY (accuracies):")
    print(f"All: {out_dict['all']['accuracy']}")
    for s in slices:
        print(f"Slice {s}: {out_dict['slice_' + s]['accuracy']}")

    return out_dict


def separate_eval_loader(data_loader):
    X = []
    Y = []
    Z = []

    # The user passes in a single data_loader and we handle splitting and
    # recombining
    for ii, data in tqdm(enumerate(data_loader), total=len(data_loader)):
        x_batch, y_batch, z_batch = data

        X.append(x_batch)
        Y.append(y_batch)
        if isinstance(z_batch, torch.Tensor):
            z_batch = z_batch.numpy()
        Z.extend([str(z) for z in z_batch])  # slice labels may be strings

    X = torch.cat(X)
    Y = torch.cat(Y)
    return X, Y, Z
