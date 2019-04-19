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
        plot_correctness(X_slice, pred_slice, gt_slice)


def visualize_attention(model, payload):
    slice_task_names = sorted(
        [slice_task_name for slice_task_name in model.slice_tasks.keys()]
    )

    Ys, A_weights = model.attention_with_gold(payload)
    A_weights = np.array(A_weights)

    X = np.array(payload.data_loader.dataset.X_dict["data"])
    for label_name, task_name in payload.labels_to_tasks.items():
        for slice_idx, head_name in enumerate(slice_task_names):
            print(f"Vizualizing {head_name} attention on {label_name}...")
            Y = np.array(Ys[label_name]).squeeze()
            slice_mask = Y != 0
            X_slice = X[slice_mask, :]
            A_slice = A_weights[slice_mask, slice_idx]  # visualize for slice_head
            plot_attention(X_slice, A_slice)


def set_and_show_plot(xlim=(-1, 1), ylim=(-1, 1)):
    # assume that all data lies within these (x, y) bounds
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.axes().set_aspect("equal")
    plt.show()


def plot_xy(X, Y, gt=None, c=None):
    Y1_mask = Y == 1
    Y2_mask = Y == 2
    plt.scatter(X[Y1_mask, 0], X[Y1_mask, 1])
    plt.scatter(X[Y2_mask, 0], X[Y2_mask, 1])
    set_and_show_plot()


def plot_correctness(X, preds, gt):
    # plot correct -> red and incorrect -> green
    correct_mask = preds == gt
    incorrect_mask = preds != gt
    plt.scatter(X[correct_mask, 0], X[correct_mask, 1], c="green")
    plt.scatter(X[incorrect_mask, 0], X[incorrect_mask, 1], c="red")
    set_and_show_plot()


def plot_attention(X, A):
    # A must be a vector; we visualize weights for 1 slice_head at a time
    assert A.shape == (len(X),)
    # plot heatmap based on c values
    sc = plt.scatter(X[:, 0], X[:, 1], c=A, vmin=0, vmax=1.0, cmap="inferno_r")
    plt.colorbar(sc)
    set_and_show_plot()
