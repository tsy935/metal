# Define custom metrics
import numpy as np
from metal.metrics import accuracy_score
from metal.utils import convert_labels

def calc_heads_acc(model, dataloader):
    L, X, Y, Z = dataloader.dataset.data
    L_probs, Y_off_probs, Y_on_probs = model.predict_all_proba(X)

    if model.indicator:
        L = L.clone()
        L[L == 2] = 1
    else:
        raise NotImplemented("Cannot evaluate L predictions if model.indicator=False")
    
    Y_off_preds = convert_labels(np.round(Y_off_probs[:, 0]), "onezero", "categorical")
    Y_on_preds = convert_labels(np.round(Y_on_probs[:, 0]), "onezero", "categorical")
    metrics = {}    
    
    metrics = {
        "L_acc": accuracy_score(L.flatten(), np.round(L_probs).flatten()),
        "Y_off_acc": accuracy_score(Y, Y_off_preds),
        "Y_on_acc": accuracy_score(Y, Y_on_preds),     
    }
    
    return metrics

def calc_slice_acc(model, dataloader):
    model.warn_once("Slice accuracies currently only work for valid set!")
    L, X, Y, Z = dataloader.dataset.data
    Y_preds = model.predict(X)

    metrics = {}
    slices = sorted(list(np.unique(Z)))
    slices.remove(0) # remove "background" points
    for s in slices:
        inds = [i for i, e in enumerate(Z) if e == s]
        X_slice = X[inds]
        Y_slice = Y[inds]
        Y_preds_slice = Y_preds[inds]
        acc = model.score((None, X_slice, Y_slice, None), verbose=False)
        metrics[f"slice_{s}"] = acc
    return metrics

