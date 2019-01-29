import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display
from scipy.sparse import csr_matrix


def display_scores(scores, x_var, x_range):
    scores = dict(scores)

    for x_val in x_range:
        df_title = f"{x_var}: {x_val}"

        pd_scores = {}
        for model_name, model_scores in scores.items():
            slice_names = model_scores[x_val][0].keys()
            mean_scores = {
                slice_name: np.mean(
                    [s[slice_name] for s in model_scores[x_val]]
                )
                for slice_name in slice_names
            }

            pd_scores[model_name] = mean_scores

        df = pd.DataFrame.from_dict(pd_scores)
        df.index.name = df_title
        display(df)


def visualize_data(X, Y, C, L):
    # show data by class
    #    plt.figure(figsize=(8, 8))
    plt.figure(figsize=(4, 4))
    #    plt.subplot(2, 2, 1)
    plt.title("Data by Class Label")
    plt.scatter(X[Y == 1, 0], X[Y == 1, 1], label="$y=1$", c="C1", s=0.2)
    plt.scatter(X[Y == 2, 0], X[Y == 2, 1], label="$y=2$", c="C0", s=0.2)
    plt.xlim(-4, 12)
    plt.ylim(-8, 8)
    plt.legend(markerscale=10, loc="upper left")
    plt.show()

    # show data by slice
    #    plt.subplot(2, 2, 2)
    plt.figure(figsize=(4, 4))
    plt.title("Data by Slice")
    for c in np.unique(C):
        plt.scatter(X[C == c, 0], X[C == c, 1], label=f"$S_{int(c)}$", s=0.2)
    plt.xlim(-4, 12)
    plt.ylim(-8, 8)
    plt.legend(markerscale=10)
    plt.show()

    # LFs targeting slices
    #    plt.subplot(2, 2, 3)
    plt.figure(figsize=(4, 4))
    plt.title("LFs ($\lambda_i$) Targeting Slices ($S_i$)")
    if isinstance(L, csr_matrix):
        L = L.toarray()
    for c in np.unique(C):
        c = int(c)
        voted_idx = np.where(L[:, c] != 0)[0]
        plt.scatter(
            X[voted_idx, 0], X[voted_idx, 1], label=f"$\lambda_{c}$", s=0.2
        )
    plt.xlim(-4, 12)
    plt.ylim(-8, 8)
    plt.legend()
    plt.show()

    #    plt.subplot(2, 2, 4)
    #    plt.title('$\lambda_2$ votes')
    #    # first plot underlying slice
    #    plt.scatter(X[C==2,0], X[C==2,1], label="$S_2$", s=0.1, c='red')
    #    plt.scatter(X[L[:,2]==1,0], X[L[:,2]==1,1], label="$\lambda_2=+1$", s=0.2, c='C1')
    #    plt.scatter(X[L[:,2]==-1,0], X[L[:,2]==-1,1], label="$\lambda_2=-1$", s=0.2, c='C0')
    #    plt.xlim(-8, 8)
    #    plt.ylim(-8, 8)
    #    plt.legend()
    # plt.show()


def plot_base_attention_delta(scores, xlabel=None):
    scores_collected = {}
    for model_name, model_scores in scores.items():
        x_range = model_scores.keys()

        # take average value across trials
        scores_collected[model_name] = [
            np.mean(np.array([s["S0"] for s in model_scores[x]]))
            for x in x_range
        ]
    delta = np.array(scores_collected["AttentionModel"]) - np.array(
        scores_collected["EndModel"]
    )

    x_range = list(scores["AttentionModel"].keys())
    plt.plot(x_range, delta)
    plt.axhline(0, linestyle="--", c="red")
    plt.title("$Attention - EndModel$ Delta Scores on S0")
    if xlabel:
        plt.xlabel(xlabel)


def plot_slice_scores(
    results,
    xlabel="Overlap Proportion",
    custom_ylims={},
    custom_xranges={},
    savedir=None,
):
    plt.figure(figsize=(10, 10))
    slice_names = ["S0", "S1", "overall"]
    for i, slice_name in enumerate(slice_names):
        plt.subplot(2, 2, i + 1)

        for model_name, model_scores in results.items():
            x_range = model_scores.keys()

            # modify x_range
            custom_xrange = custom_xranges.get(slice_name, None)
            if custom_xrange:
                x_range = custom_xrange

            # take average value across trials
            scores_collected = [
                np.mean(np.array([s[slice_name] for s in model_scores[x]]))
                for x in x_range
            ]

            plt.plot(x_range, scores_collected, label=model_name)

        # print x-axis in precision 2
        x_range = ["%.2f" % float(x) for x in x_range]

        plt.title(f"Accuracy on {slice_name} vs. {xlabel}")
        plt.xlabel(xlabel)
        plt.ylabel(f"Accuracy on {slice_name}")

        # modify ylim
        custom_ylim = custom_ylims.get(slice_name, None)
        if custom_ylim:
            plt.ylim(bottom=custom_ylim[0], top=custom_ylim[1])
        else:
            plt.ylim(bottom=0, top=1)
        plt.legend()

    plt.subplot(2, 2, 4)
    plot_base_attention_delta(results, xlabel)

    plt.show()

    if savedir:
        plt.savefig(os.path.join(savedir, "results.png"))


def plot_predictions(X_test, Y_test, model, C=None):
    Y_p, Y = model._get_predictions((X_test, Y_test))

    # correct_idx = np.where(Y_p == Y)[0]
    wrong_idx = np.where(Y_p != Y)[0]
    if C is not None:
        for c in np.unique(C):
            c_idx = np.where(C == c)[0]
            plt.scatter(X_test[c_idx, 0], X_test[c_idx, 1], s=3)
    #    plt.scatter(X_test[correct_idx, 0], X_test[correct_idx, 1], c='grey', label='correct', s=7)
    plt.scatter(
        X_test[wrong_idx, 0],
        X_test[wrong_idx, 1],
        c="red",
        label="incorrect prediction",
        s=10,
    )
    plt.legend(markerscale=1)
    plt.xlim(-4, 4)
    plt.ylim(-4, 4)


def compare_prediction_plots(test_data, trained_models, C=None):
    X_test, Y_test = test_data

    num_plots = len(trained_models) + 1  # +1 for GT plot
    plt.figure(figsize=(4 * num_plots, 4))
    plt.subplot(1, num_plots, 1)
    plt.title("GT Classes")
    class_1_idx = np.where(Y_test == 1)[0]
    class_2_idx = np.where(Y_test == 2)[0]
    plt.scatter(
        X_test[class_1_idx, 0],
        X_test[class_1_idx, 1],
        label="$y=1$",
        c="C1",
        s=2,
    )
    plt.scatter(
        X_test[class_2_idx, 0],
        X_test[class_2_idx, 1],
        label="$y=2$",
        c="C0",
        s=2,
    )
    plt.xlim(-4, 12)
    plt.ylim(-8, 8)
    plt.legend()

    for i, (model_name, model) in enumerate(trained_models.items()):
        #        plt.subplot(1,num_plots,i+2)
        plt.figure(figsize=(4, 4))
        plt.title(model_name)
        plot_predictions(X_test, Y_test, model, C)
        plt.show()
    plt.show()
