import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from scipy.sparse import csr_matrix

from metal.contrib.slicing.utils import generate_weak_labels
from metal.end_model import EndModel
from metal.contrib.slicing.online_dp import SliceHatModel
from metal.utils import SlicingDataset

def create_data_loaders(Ls, Xs, Ys, Zs, model_config):
    """
    Creates train, dev, and test dataloaders based on raw input data and config.

    Returns:
        (train_dl, dev_dl, test_dl)
    """
    
    L_weights = model_config.get("L_weights")
    assert(isinstance(L_weights, list) or L_weights is None)

    is_slicing = "slice_kwargs" in model_config.keys()
    
    # Generate weak labels:
    # a) uniform (L_weights = [1,...,1])
    # b) manual  (L_weights = [1,X,...1])
    # c) learned (L_weights = None): DP
    L_train = Ls[0].toarray() if isinstance(Ls[0], csr_matrix) else Ls[0]
    Y_weak = generate_weak_labels(L_train, L_weights)

    if is_slicing:
        train_dataset = SlicingDataset(Xs[0], L_train, Y_weak)
    else:
        train_dataset = SlicingDataset(Xs[0], Y_weak)

    dev_dataset = SlicingDataset(Xs[1], Ys[1])
    test_dataset = SlicingDataset(Xs[2], Ys[2], Zs[2])

    batch_size = model_config.get("train_kwargs", {}).get("batch_size", 32)
    return (
        DataLoader(train_dataset, batch_size=batch_size, shuffle=True), 
        DataLoader(dev_dataset, batch_size=batch_size, shuffle=False), 
        DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    )


def train_model(
    config,
    train_loader,
    dev_loader,
    m,
    verbose=False):
    """
    Generates weak labels and trains a single model

    Args:
        config: model config with "end_model_init_kwargs"
            optional: "train_kwargs", "slice_kwargs" (if slice model)
        train_loader: data loader for train data 
            if slice: (X, L, Y) 
            else: (X, Y)
        dev_loader: dataloader for validation data (X, Y)
        m: num labeling sources

    Returns:
        model: a trained model
    """

    # Instantiate end model
    model = EndModel(**config["end_model_init_kwargs"])

    # Add slice hat if applicable
    slice_kwargs = config.get('slice_kwargs')
    if slice_kwargs:
        model = SliceHatModel(model, m, **slice_kwargs)

    # train model
    train_kwargs = config.get("train_kwargs", {})
    train_kwargs["disable_prog_bar"] = True
    model.train_model(train_loader, dev_data=dev_loader, **train_kwargs)

    return model


def eval_model(
    model,
    eval_loader,
    metrics=["accuracy", "precision", "recall", "f1"],
    verbose=True,
    break_ties="random",
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
            (X_slice, Y_slice), metrics, verbose=verbose, break_ties=break_ties
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
