import matplotlib.pyplot as plt
import numpy as np


def visualize_payload(payload):
    """ Visualizes all labelsets for a given payload. """
    X = np.array(payload.data_loader.dataset.X_dict["data"])
    labelsets = payload.data_loader.dataset.Y_dict
    for label_name, labels in labelsets.items():
        print(f"Vizualizing {label_name} from {payload.name}...")
        Y = labels.numpy()
        plot_xy(X, Y)


def visualize_predictions(model, payload):
    """ Use model to evaluate on payload and visualize
    correct/incorrect predictions """
    target_tasks = list(payload.labels_to_tasks.values())
    target_labels = list(payload.labels_to_tasks.keys())
    Ys, Ys_probs, Ys_preds = model.predict_with_gold(
        payload, target_tasks, target_labels, return_preds=True
    )

    X = np.array(payload.data_loader.dataset.X_dict["data"])
    for label_name, task_name in payload.labels_to_tasks.items():
        model_name = model.__class__.__name__
        print(f"Vizualizing {model_name} predictions on {label_name}...")
        Y = np.array(Ys[label_name]).squeeze()
        preds = np.array(Ys_preds[task_name]).squeeze()

        slice_mask = Y != 0
        X_slice = X[slice_mask, :]
        pred_slice = preds[slice_mask]
        gt_slice = Y[slice_mask]
        plot_xy(X_slice, pred_slice, gt_slice)


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
    plt.axes().set_aspect("equal")
    plt.show()
