from collections import defaultdict

import numpy as np

from metal.contrib.backends.snorkel_gm_wrapper import SnorkelLabelModel
from metal.contrib.slicing.experiment_utils import (
    create_data_loader,
    eval_model,
    parse_history,
    search_upweighting_models,
    train_model,
)
from metal.contrib.slicing.mixture_of_experts import train_MoE_model
from metal.contrib.slicing.synthetics.geometric_synthetics import (
    generate_dataset,
)
from metal.label_model import MajorityLabelVoter
from metal.utils import split_data

# NOTE: each model can take a "train_kwargs"

# SHARED PIECES
end_model_init_kwargs = {"layer_out_dims": [2, 10, 10, 2], "verbose": False}

# FULL CONFIGS
dp_config = {"end_model_init_kwargs": end_model_init_kwargs}

uni_config = {"end_model_init_kwargs": end_model_init_kwargs}

up_config = {
    "end_model_init_kwargs": end_model_init_kwargs,
    "upweight_search_space": {"range": [1, 5]},
    "max_search": 5,
}

moe_config = {
    "end_model_init_kwargs": end_model_init_kwargs,
    "expert_train_kwargs": {"n_epochs": 10, "verbose": False},
    "train_kwargs": {"verbose": False},
    "gating_dim": 5,
}

sm_config = {
    "end_model_init_kwargs": end_model_init_kwargs,
    "slice_kwargs": {"slice_weight": 0.1},
}


model_configs = {
    #     "UNI": uni_config,
    #     "UPx2": up_config,
    #     "MoE": moe_config,
    "DP": dp_config,
    "SM": sm_config,
}

for ITER in [1, 2, 3, 4, 5]:
    NUM_TRIALS = 10
    NUM_SLICES = 5
    K = 2
    M = 20
    N = 5000
    salt = (
        42 + ITER
    )  # A base to add to trial number to set a unique seed for each trial

    history = defaultdict(list)
    for trial in range(NUM_TRIALS):
        print(f"[Trial {trial}]")

        Z_kwargs = {"num_slices": NUM_SLICES}
        L, X, Y, Z, targeting_lfs_idx = generate_dataset(
            K,
            M,
            N,
            Z_kwargs=Z_kwargs,
            return_targeting_lfs=True,
            plotting=False,
            seed=(salt + trial),
        )

        Ls, Xs, Ys, Zs = split_data(
            L, X, Y, Z, splits=[0.5, 0.25, 0.25], shuffle=True
        )

        for model_name, model_config in model_configs.items():
            print("-" * 10, "Training", model_name, "-" * 10)

            # Generate weak labels:
            if model_name == "UNI" or model_name.startswith("UP"):
                Y_train = MajorityLabelVoter().predict_proba(Ls[0])
            else:
                label_model = SnorkelLabelModel()
                label_model.train_model(Ls[0])
                Y_train = label_model.predict_proba(Ls[0])
            Ys[0] = Y_train

            # Train end model

            if model_name == "UNI":
                L_weights = list(np.ones(M))
                model = train_model(model_config, Ls, Xs, Ys, Zs, L_weights)
            elif model_name.startswith("UP"):
                model = search_upweighting_models(
                    model_config,
                    Ls,
                    Xs,
                    Ys,
                    Zs,
                    targeting_lfs_idx,
                    verbose=False,
                )
            elif model_name == "MoE":
                model = train_MoE_model(model_config, Ls, Xs, Ys, Zs)
            elif model_name == "DP":
                model = train_model(model_config, Ls, Xs, Ys, Zs)
            elif model_name == "SM":
                model = train_model(model_config, Ls, Xs, Ys, Zs)
            else:
                raise Exception(f"Unrecognized model_name: {model_name}")

            test_loader = create_data_loader(
                Ls, Xs, Ys, Zs, model_config, "test"
            )
            results = eval_model(
                model, test_loader, verbose=False, summary=False
            )

            # Save results
            history[model_name].append(results)

    print(f"ITER: {ITER}")
    print(f"Average (n={NUM_TRIALS}):")
    df = parse_history(history, NUM_SLICES)
    print(df)
