"""Runs simulations over equal weights, manual reweighting,
and attention based models.

Sample command:
python simulate.py --var cov --save-dir results/test --n 50 --x-range 0.6 0.7 0.8 0.9 1.0
"""

import argparse
import json
import os
import random
import sys
from collections import defaultdict
from time import strftime, time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from synthetics_utils import generate_synthetic_data
from visualization_utils import plot_slice_scores, visualize_data, compare_prediction_plots

from metal.contrib.logging.tensorboard import TensorBoardWriter
from metal.contrib.slicing.experiment_utils import (
    generate_weak_labels,
    compute_lf_accuracies,
    get_weighted_sampler_via_targeting_lfs
)
from metal.contrib.slicing.online_dp import (
    LinearModule,
    MLPModule,
    SliceDPModel,
)
from metal.end_model import EndModel


# Import tqdm_notebook if in Jupyter notebook
try:
    from IPython import get_ipython

    if "IPKernelApp" not in get_ipython().config:
        raise ImportError("console")
except (AttributeError, ImportError):
    from tqdm import tqdm
else:
    from tqdm import tqdm_notebook as tqdm


sys.path.append("/dfs/scratch0/vschen/metal")


# Define default simulation configs; can be overwritten
data_config = {
    # data generation
    "N": 10000,  # num data points
    "mus": [
        np.array([-8, 0]),  # Mode 1: Y = -1
        np.array([3, 0]),  # Mode 2: Y = 1
    ],
    "labels": [-1, 1],  # labels of each slice
    "props": [0.25, 0.75],  # proportion of data in each mode
    "variances": [3, 5],  # proportion of data in each mode
    "head_config": {
        "h": 5,  # horizontal shift of slice
        "k": -2.5,  # vertical shift of slice
        "r": 1.,  # radius of slice
        "slice_label": -1,
    },
    "accs": np.array([0.95, 0.95, 0.95]),  # default accuracy of LFs
    "covs": [
        ("recall", 0.95),
        ("recall", 0.95),
        ("recall", 0.95),
    ],  # coverage of LFs, as define by prec., rec.
}

experiment_config = {
    "num_trials": 1,
    "x_range": np.linspace(0, 1.0, 5),
    "x_var": None,
    "input_module_class": MLPModule,
    "input_module_kwargs": {
        "input_dim": 2,
        "middle_dims": [10, 10],
        "bias": True,
    },
    "train_kwargs": {
        "n_epochs": 20,
        "print_every": 10,
        "validation_metric": "accuracy",
        "disable_prog_bar": True,
        "lr": 0.005,
        "checkpoint_runway": 5,
    },
    "verbose": False,
    "train_prop": 0.7,
    "dev_prop": 0.1,
    "use_weak_labels_from_gen_model": False,
    "tensorboard_logdir": "./run_logs",
    "seed": False,
}

model_configs = {
    "EndModel": {
        "base_model_class": EndModel,
        "input_module_class": MLPModule,
        "input_module_init_kwargs": {
            "input_dim": 2,
            "middle_dims": [10, 10],
            "bias": True,
            "output_dim": 10,
        },
        "base_model_init_kwargs": {
            "layer_out_dims": [10, 2],
            "input_layer_config": {
                "input_relu": False,
                "input_batchnorm": False,
                "input_dropout": 0.0,
            },
        },
        "train_on_L": False,
    },
    "UniformModel": {
        "base_model_class": SliceDPModel,
        "base_model_init_kwargs": {
            "reweight": False,
            "r": 10,
            "slice_weight": 0.5,
            "L_weights": np.array([1., 1., 1.]).astype(np.float32),
        },
        "input_module_class": MLPModule,
        "input_module_init_kwargs": {
            "input_dim": 2,
            "middle_dims": [10, 10],
            "bias": True,
            "output_dim": 10,
        },
        "train_on_L": True,
    },
    "ManualModel": {
        "base_model_class": SliceDPModel,
        "base_model_init_kwargs": {
            "reweight": False,
            "r": 10,
            "slice_weight": 0.5,
            "L_weights": np.array([1., 1., 5.]).astype(
                np.float32
            ),  # LF2 w/ 5x weight
        },
        "input_module_class": MLPModule,
        "input_module_init_kwargs": {
            "input_dim": 2,
            "middle_dims": [10, 10],
            "bias": True,
            "output_dim": 10,
        },
        "train_on_L": True,
    },
    "AttentionModel": {
        "base_model_class": SliceDPModel,
        "base_model_init_kwargs": {
            "reweight": True,
            "r": 10,
            "slice_weight": 0.5,
            "L_weights": None,
        },
        "input_module_class": MLPModule,
        "input_module_init_kwargs": {
            "input_dim": 2,
            "middle_dims": [10, 10],
            "bias": True,
            "output_dim": 10,
        },
        "train_on_L": True,
    },
}


class SyntheticDataset(torch.utils.data.Dataset):
    def __init__(self, X, Y, L=None):
        self.X = X
        if len(Y.shape) == 1:
            print ("Y is one dimensional. Converting to cat labels")
            Y_cat = np.zeros((Y.shape[0], 2))
            Y_cat[:,0] = Y == 1
            Y_cat[:,1] = Y == 2
            self.Y = Y_cat.astype(np.float32)
        else:
            self.Y = Y.astype(np.float32)

        self.L = L

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        if self.L is None:
            return self.X[idx], self.Y[idx]
        else:
            return self.X[idx], self.L[idx], self.Y[idx]


def train_models(
    X_train,
    L_train,
    Y_train,
    test_data,
    model_configs,
    train_kwargs,
    verbose=False,
    use_cuda=False,
    seed=123,
):
    """
    Trains baseline, oracle, and attention model
    Args:
        - train_data: (X, L) data for slice model to train on
        - X_train: [N, d] train data of dimension d
        - L_train: [N, m] label matrix over train data generated by m LFs
        - Y_train: [N, 1or2] weak or gt labels, can be 1d or categorical
        - test_data: (X, Y) data for validation
        - model_configs: dictionary mapping model_name:init/train_config
        - train_kwargs: kwargs to be passed to each model for training
    Returns:
        - dict of {model_name: trained_model}
    """
    trained_models = {}
    for model_name, config in model_configs.items():
        if verbose:
            print("-" * 10, f"Training {model_name}", "-" * 10)
        base_model_class = config["base_model_class"]
        base_model_init_kwargs = config["base_model_init_kwargs"]
        base_model_init_kwargs.update({"m": L_train.shape[1]}) # get m from L dim
        input_module_class = config["input_module_class"]
        input_module_init_kwargs = config["input_module_init_kwargs"]

        # init base model (i.e. EndModel or SliceDPModel)
        model = base_model_class(
            input_module=input_module_class(**input_module_init_kwargs),
            **base_model_init_kwargs,
            verbose=verbose,
            use_cuda=use_cuda,
            seed=seed,
        )

        # create dataloader
        train_dataset = (
            SyntheticDataset(X_train, Y_train, L_train) if config["train_on_L"]
            else SyntheticDataset(X_train, Y_train)
        )
        sampler = None
        multipliers = config.get("upsample_lf0", None)
        if multipliers is not None:
            sampler = get_weighted_sampler_via_targeting_lfs(L_train, [0], 3)

        train_loader = DataLoader(
            train_dataset, 
            batch_size=train_kwargs.get("batch_size", 32),
            num_workers=1,
            sampler=sampler
        )
        # train model
        model.train_model(
            train_loader,
            dev_data=test_data,
            **train_kwargs,
        )

        # collect trained models in dict
        trained_models[model_name] = model

    return trained_models

def eval_model(model, data, eval_dict):
    """Evaluates models according to indexes in 'eval_dict'
    Args:
        model: trained model to evaluate
        data: (X,Y) full test set to evaluate on
        eval_dict: mapping eval slice {"slice_name":idx}
            where idx is list of indexes for corresponding slice
    Returns:
        results_dict: mapping {"slice_name": scores}
            includes "overall" accuracy by default
    """
    slice_scores = {}
    for slice_name, eval_idx in eval_dict.items():
        slice_scores[slice_name] = model.score_on_slice(
            data, eval_idx, metric="accuracy", verbose=False
        )

    slice_scores["overall"] = model.score(
        data, metric="accuracy", verbose=False
    )
    return slice_scores


def simulate(data_config, generate_data_fn, experiment_config, model_configs):
    """Simulates training over data (specified by data_config) with models specified
    in model_configs over the specified config, varying values specified in experiment_config.

    Args:
        config: for data generation
        generate_data_fn: data generation fn that accepts config, x_var, x_val
            and returns (X, Y, C, L)
    Returns: dict with simulation scores in format:
        {model_name: {parameter_name: [score_for_trial_i]}}
    """
    if experiment_config.get("seed", False):
        np.random.seed(experiment_config["seed"])
        random.seed(experiment_config["seed"])

    # to collect scores for all models
    scores = defaultdict(
        lambda: defaultdict(list)
    )  # {model_name: x_val: list of scores}

    # get config variables
    num_trials = experiment_config["num_trials"]
    x_range = experiment_config["x_range"]
    var_name = experiment_config["x_var"]
    if var_name is None:
        x_range = [None]

    # for each value, run num_trials simulations
    for x in x_range:
        print(f"Simulating: {var_name}={x}")
        for _ in tqdm(range(num_trials)):

            # generate data
            X, Y, C, L = generate_data_fn(
                data_config, var_name, x, verbose=experiment_config["verbose"]
            )
            if experiment_config.get("visualize_data", False):
                visualize_data(X, Y, C, L)

            # convert to multiclass labels
            if -1 in Y:
                raise ValueError("No -1 labels allowed")

            # create data splits
            X = torch.from_numpy(X.astype(np.float32))
            L = L.astype(np.float32)
            Y = Y.astype(np.float32)

            train_end_idx = int(len(X) * experiment_config["train_prop"])
            dev_end_idx = train_end_idx + int(len(X) * experiment_config["dev_prop"])

            X_train, X_dev, X_test = X[:train_end_idx], X[train_end_idx:dev_end_idx], X[dev_end_idx:]
            Y_train, Y_dev, Y_test = Y[:train_end_idx], Y[train_end_idx:dev_end_idx], Y[dev_end_idx:]
            C_train, C_dev, C_test = C[:train_end_idx], C[train_end_idx:dev_end_idx], C[dev_end_idx:]
            L_train, L_dev, L_test = L[:train_end_idx], L[train_end_idx:dev_end_idx], L[dev_end_idx:]

            test_data = (X_test, Y_test)

            # generate weak labels
            if experiment_config["use_weak_labels_from_gen_model"]:
                Y_weak = generate_weak_labels(
                    L_train, verbose=experiment_config["verbose"]
                )
            else:
                accs = compute_lf_accuracies(L_dev, Y_dev)
                Y_weak = generate_weak_labels(
                    L_train,
                    accs,
                    verbose=experiment_config["verbose"],
                )

            trained_models = train_models(
                X_train,
                L_train,
                Y_weak,
                test_data,
                model_configs,
                experiment_config["train_kwargs"],
                verbose=experiment_config["verbose"],
                seed=experiment_config.get("seed", None),
            )

            # score the models
            eval_dict = {
                "S0": np.where(C_test == 0)[0], 
                "S1": np.where(C_test == 1)[0]
            }
            for model_name, model in trained_models.items():
                scores[model_name][x].append(
                    eval_model(model, test_data, eval_dict)
                )

            if experiment_config["plot_predictions"]:
                model1, model2 = tuple(experiment_config["plot_predictions"])
                compare_prediction_plots((X_test, Y_test), trained_models[model1], trained_models[model2])

    return scores


if __name__ == "__main__":
    import warnings

    warnings.filterwarnings("ignore")
    # TODO: fix warnings

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--variable",
        choices=["sp", "acc", "cov.recall", "cov.precision"],
        help="variable we are varying in simulation",
    )
    parser.add_argument("--save-dir", type=str, help="where to save results")
    parser.add_argument(
        "--n", type=int, default=25, help="num trials to run of simulation"
    )
    parser.add_argument(
        "--x-range",
        type=float,
        nargs="+",
        default=None,
        help="range of values to scan over",
    )
    parser.add_argument(
        "--radius", type=float, default=None, help="radius of slice head"
    )
    args = parser.parse_args()

    # override simulation config
    if args.n:
        experiment_config["num_trials"] = args.n
    if args.x_range:
        experiment_config["x_range"] = args.x_range
    if args.x_range:
        experiment_config["x_var"] = args.variable
    if args.save_dir:
        experiment_config["tensorboard_logdir"] = args.save_dir
    if args.radius:
        data_config["head_config"]["r"] = args.radius

    # run simulations
    results = simulate(
        data_config, generate_synthetic_data, experiment_config, model_configs
    )

    # save scores and plot
    print(f"Saving to {args.save_dir}")
    results_path = os.path.join(args.save_dir, f"{args.variable}-results.json")
    os.makedirs(args.save_dir, exist_ok=True)
    json.dump(results, open(results_path, "w"))
    if args.variable == "sp":
        xlabel = "Slice Proportion"
    elif args.variable == "acc":
        xlabel = "Head Accuracy"
    elif args.variable == "cov.recall":
        xlabel = "Head Recall"
    elif args.variable == "cov.precision":
        xlabel = "Head Precision"

    plot_slice_scores(results, xlabel=xlabel, savedir=args.save_dir)
