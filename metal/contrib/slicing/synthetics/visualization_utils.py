import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display


def display_scores(scores, x_var, x_range):
    scores = dict(scores)

    for x_val in x_range:
        df_title = f"{x_var}: {x_val}"

        pd_scores = {}
        for model_name, model_scores in scores.items():
            slice_names = model_scores[x_val][0].keys()
            mean_scores = {slice_name : np.mean([s[slice_name] for s in model_scores[x_val]]) 
                            for slice_name in slice_names}

            pd_scores[model_name] = mean_scores

        df = pd.DataFrame.from_dict(pd_scores)
        df.index.name = df_title
        display(df)


def visualize_data(X, Y, C, L):
    # show data by class
    plt.figure(figsize=(8, 8))
    plt.subplot(2, 2, 1)
    plt.title('Data by Classes')
    plt.scatter(X[Y==1,0], X[Y==1,1], label="$y=+1$", c='C1', s=0.2)
    plt.scatter(X[Y==-1,0], X[Y==-1,1], label="$y=-1$", c='C0', s=0.2)
    plt.xlim(-8, 8)
    plt.ylim(-8, 8)
    plt.legend()

    # show data by slice
    plt.subplot(2, 2, 2)
    plt.title("Data by Slice")
    for c in np.unique(C):
        plt.scatter(X[C==c,0], X[C==c,1], label=f"$S_{int(c)}$", s=0.2)
    plt.xlim(-8, 8)
    plt.ylim(-8, 8)
    plt.legend()

    # LFs targeting slices
    plt.subplot(2, 2, 3)
    plt.title("LFs ($\lambda_i$) Targeting Slices ($S_i$)")
    for c in np.unique(C):
       c = int(c)
       plt.scatter(X[L[:,c]!=0,0], X[L[:,c]!=0,1], label=f"$\lambda_{c}$", s=0.2)
    plt.xlim(-8, 8)
    plt.ylim(-8, 8)
    plt.legend()
    
#    plt.subplot(2, 2, 4)
#    plt.title('$\lambda_2$ votes')
#    # first plot underlying slice
#    plt.scatter(X[C==2,0], X[C==2,1], label="$S_2$", s=0.1, c='red')
#    plt.scatter(X[L[:,2]==1,0], X[L[:,2]==1,1], label="$\lambda_2=+1$", s=0.2, c='C1')
#    plt.scatter(X[L[:,2]==-1,0], X[L[:,2]==-1,1], label="$\lambda_2=-1$", s=0.2, c='C0')    
#    plt.xlim(-8, 8)
#    plt.ylim(-8, 8)
#    plt.legend()
    plt.show()


def plot_slice_scores(
    results,
    xlabel="Overlap Proportion",
    custom_ylims={},
    custom_xranges={},
    savedir=None,
):
    plt.figure(figsize=(10, 10))
    slice_names = ["S0", "S1", "S2", "overall"]
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
    plt.show()

    if savedir:
        plt.savefig(os.path.join(savedir, "results.png"))
