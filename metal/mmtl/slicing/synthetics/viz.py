import matplotlib.pyplot as plt
import numpy as np


def visualize_payload(p):
    X = np.array(p.data_loader.dataset.X_dict["data"])
    labelsets = p.data_loader.dataset.Y_dict
    for label_name, labels in labelsets.items():
        print(f"Vizualizing {label_name} from {p.name}...")
        Y = labels.numpy()
        plot_xy(X, Y)


def plot_xy(X, Y, gt=None):
    """ Plot synthetic data.
    If gt is provided, treats 'Y' as predictions, and shows correct/incorrect in
    green/red.

    If gt is not provided, treats Y as gt, and shows Y=1 vs. Y=2
    """
    if gt is not None:
        preds = Y
        correct_mask = preds == gt
        incorrect_mask = preds != gt
        plt.scatter(X[correct_mask, 0], X[correct_mask, 1], c="green")
        plt.scatter(X[incorrect_mask, 0], X[incorrect_mask, 1], c="red")

    else:
        Y1_mask = Y == 1
        Y2_mask = Y == 2
        plt.scatter(X[Y1_mask, 0], X[Y1_mask, 1])
        plt.scatter(X[Y2_mask, 0], X[Y2_mask, 1])

    # assume that all data lies within these (x, y) bounds
    plt.xlim([-1, 1])
    plt.ylim([-1, 1])
    plt.show()
