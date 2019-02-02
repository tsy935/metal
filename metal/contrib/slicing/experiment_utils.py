from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from metal.contrib.slicing.online_dp import (
    SliceDPModel,
    SliceHatModel,
    SliceOnlineModel,
)
from metal.contrib.slicing.slice_model import SliceMaster
from metal.contrib.slicing.utils import get_L_weights_from_targeting_lfs_idx
from metal.end_model import EndModel
from metal.label_model.baselines import WeightedLabelVoter
from metal.tuners.tuner import ModelTuner
from metal.utils import SlicingDataset


def create_data_loader(Ls, Xs, Ys, Zs, model_config, split):
    """
    Creates train, dev, or test dataloaders based on raw input data and config.

    Returns:
        (train_dl, dev_dl, test_dl)
    """
    assert split in ["train", "dev", "test"]

    is_slicing = "slice_kwargs" in model_config.keys()
    Ls_tensors = [torch.Tensor(np.asarray(L.todense())) for L in Ls]

    if split == "train":
        dataset = SlicingDataset(Ls_tensors[0], Xs[0], Ys[0], Zs[0])
        shuffle = True

    elif split == "dev":
        dataset = SlicingDataset(Ls_tensors[1], Xs[1], Ys[1], Zs[1])
        shuffle = False

    elif split == "test":
        dataset = SlicingDataset(Ls_tensors[2], Xs[2], Ys[2], Zs[2])
        shuffle = False

    batch_size = model_config.get("train_kwargs", {}).get("batch_size", 32)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def train_model(config, Ls, Xs, Ys, Zs, L_weights=None, model_key=None):
    """
    Generates weak labels and trains a single model

    Returns:
        model: a trained model
    """

    # Instantiate end model
    model = EndModel(**config["end_model_init_kwargs"])

    # Add slice hat if applicable
    slice_kwargs = config.get("slice_kwargs")
    if slice_kwargs:
        m = Ls[0].shape[1]  # number of LFs

        if model_key == "hat":
            model = SliceHatModel(model, m, **slice_kwargs)
        elif model_key == "online":
            model = SliceOnlineModel(model, m, **slice_kwargs)
        else:
            print (f"Default to {SliceMaster.__name__}")
            model = SliceMaster(model, m, **slice_kwargs)

    # Create data loaders
    train_loader = create_data_loader(Ls, Xs, Ys, Zs, config, "train")
    dev_loader = create_data_loader(Ls, Xs, Ys, Zs, config, "dev")

    # train model
    train_kwargs = config.get("train_kwargs", {})
    model.train_model(train_loader, valid_data=dev_loader, **train_kwargs)

    return model


def train_slice_dp(config, Ls, Xs, Ys, Zs):
    """ Trains "original" SliceDPModel """
    Ls_tensors = [torch.Tensor(np.asarray(L.todense())) for L in Ls]

    # convert to categorical labels, this usually happens with GT
    if len(Ys[0].shape) == 1:
        Y_train = Ys[0].copy()
        Y_train[Y_train == 2] = 0
        Y_train = np.expand_dims(Y_train, axis=1)
        Y_train = np.concatenate((Y_train, 1 - Y_train), axis=1)
        Y_train = torch.Tensor(Y_train)
        Ys[0] = Y_train

    batch_size = config.get("train_kwargs", {}).get("batch_size", 32)
    train_loader = DataLoader(
        SlicingDataset(Xs[0], Ls_tensors[0], Ys[0]),
        batch_size=batch_size,
        shuffle=True,
    )
    dev_loader = DataLoader(
        SlicingDataset(Xs[1], torch.Tensor(Ys[1])), batch_size=batch_size
    )

    n, m = Ls[0].shape
    input_module = config["input_module_class"](
        **config["input_module_init_kwargs"]
    )
    slice_orig = SliceDPModel(input_module, m=m, **config["slice_kwargs"])

    slice_orig.train_model(
        train_loader, valid_data=dev_loader, **config["train_kwargs"]
    )

    return slice_orig


def eval_model(
    model,
    eval_loader,
    metrics=["accuracy"],
    verbose=True,
    summary=True,
    break_ties="random",
):
    """
    Args:
        model: a trained EndModel (or subclass)
        eval_loader: a loader containing X, Y, Z
    """
    L, X, Y, Z = separate_eval_loader(eval_loader)
    out_dict = {}

    # Evaluating on full dataset
    if verbose:
        print(f"All: {len(Z)} examples")
    scores = model.score((None, X, Y, None), metrics, verbose=verbose)
    out_dict["all"] = {metrics[i]: scores[i] for i in range(len(metrics))}

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
            (None, X_slice, Y_slice, None),
            metrics,
            verbose=verbose,
            break_ties=break_ties,
        )

        out_dict[f"slice_{s}"] = {
            metrics[i]: metrics_slice[i] for i in range(len(metrics_slice))
        }

    if summary:
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
    for ii, data in enumerate(data_loader):
        _, x_batch, y_batch, z_batch = data

        X.append(x_batch)
        Y.append(y_batch)
        if isinstance(z_batch, torch.Tensor):
            z_batch = z_batch.numpy()
        Z.extend([str(z) for z in z_batch])  # slice labels may be strings

    X = torch.cat(X)
    Y = torch.cat(Y)
    return _, X, Y, Z


def search_upweighting_models(
    config, Ls, Xs, Ys, Zs, targeting_lfs_idx, verbose=False
):

    # init model
    model = EndModel(**config["end_model_init_kwargs"])
    search_space = config["upweight_search_space"]
    max_search = config.get("max_search")
    m = Ls[0].shape[1]  # number of LFs

    # initialize datasets
    dev_loader = create_data_loader(Ls, Xs, Ys, Zs, config, "dev")

    # generate L_weight multipliers based on config search space
    best_model = None
    best_score = -1
    for search_config in ModelTuner.config_generator(
        {"multiplier": search_space}, max_search
    ):
        # upweight label matrix at LFs targeting the slice
        L_weights = get_L_weights_from_targeting_lfs_idx(
            m, targeting_lfs_idx, search_config["multiplier"]
        )

        Y_weak = WeightedLabelVoter(L_weights).predict_proba(Ls[0])
        Ys[0] = Y_weak

        train_loader = create_data_loader(Ls, Xs, Ys, Zs, config, "train")

        train_kwargs = config.get("train_kwargs", {})
        model.train_model(train_loader, valid_data=dev_loader, **train_kwargs)
        score = model.score(dev_loader, verbose=verbose)
        if score > best_score:
            # TODO: save model with best slice-specific scores.. but need to define which slice
            if verbose:
                print(
                    f"Saving model with L_weight multiplier {search_config['multiplier']}"
                )
            best_score = score
            best_model = model

    return best_model


def parse_history(history, num_slices):
    REPORTING_GROUPS = ["all"] + [f"slice_{s}" for s in range(num_slices + 1)]
    METRIC_NAME = "accuracy"

    model_scores_by_slice = defaultdict(dict)
    for model_name, model_scores in history.items():
        for slice_name in REPORTING_GROUPS:
            slice_scores = [
                run[slice_name][METRIC_NAME] for run in model_scores
            ]
            mean_slice_score = sum(slice_scores) / len(slice_scores)
            model_scores_by_slice[model_name][slice_name] = mean_slice_score

    # Calculate average slice score
    for model, scores in model_scores_by_slice.items():
        slice_scores = [
            score
            for slice, score in scores.items()
            # average all slices but not the background
            if slice.startswith("slice") and slice is not "slice_0"
        ]
        model_scores_by_slice[model]["slice_avg"] = np.mean(slice_scores)

    df = pd.DataFrame.from_dict(model_scores_by_slice)
    return df
