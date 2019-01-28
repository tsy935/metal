import numpy as np
from scipy.special import expit

from metal.label_model.label_model import LabelModel
from metal.utils import recursive_merge_dicts
from metal.label_model.lm_defaults import lm_default_config


class RandomVoter(LabelModel):
    """
    A class that votes randomly among the available labels
    """

    def train_model(self, *args, **kwargs):
        pass

    def predict_proba(self, L):
        """
        Args:
            L: An [n, m] scipy.sparse matrix of labels
        Returns:
            output: A [n, k] np.ndarray of soft predictions
        """
        n = L.shape[0]
        Y_p = np.random.rand(n, self.k)
        Y_p /= Y_p.sum(axis=1).reshape(-1, 1)
        return Y_p


class MajorityClassVoter(RandomVoter):
    """
    A class that places all probability on the majority class based on class
    balance (and ignoring the label matrix).

    Note that in the case of ties, non-integer probabilities are possible.
    """

    def train_model(self, balance, *args, **kwargs):
        """
        Args:
            balance: A 1d arraylike that sums to 1, corresponding to the
                (possibly estimated) class balance.
        """
        self.balance = np.array(balance)

    def predict_proba(self, L):
        n = L.shape[0]
        Y_p = np.zeros((n, self.k))
        max_classes = np.where(self.balance == max(self.balance))
        for c in max_classes:
            Y_p[:, c] = 1.0
        Y_p /= Y_p.sum(axis=1).reshape(-1, 1)
        return Y_p


class MajorityLabelVoter(RandomVoter):
    """
    A class that places all probability on the majority label from all
    non-abstaining LFs for that task.

    Note that in the case of ties, non-integer probabilities are possible.
    """

    def train_model(self, *args, **kwargs):
        pass

    def predict_proba(self, L):
        L = self._to_numpy(L).astype(int)
        n, m = L.shape
        Y_p = np.zeros((n, self.k))
        for i in range(n):
            counts = np.zeros(self.k)
            for j in range(m):
                if L[i, j]:
                    counts[L[i, j] - 1] += 1
            Y_p[i, :] = np.where(counts == max(counts), 1, 0)
        Y_p /= Y_p.sum(axis=1).reshape(-1, 1)
        return Y_p


class WeightedLabelVoter(LabelModel):
    """
    Combines label matrix using pre-defined weights.
    """
    def __init__(self, weights, **kwargs):
        self.weights = np.array(weights)
        config = recursive_merge_dicts(lm_default_config, kwargs)
        super().__init__(k=2, config=config)

    def train_model(self, *args, **kwargs):
        pass

    def predict_proba(self, L):
        print (f"Warning: {self.__class__.__name__} only accepts k=2 class L_matrix.")
        L_np = L.copy()

        weights = self.weights
        if np.any(weights >= 1):
            weights = weights / np.max(
                weights + 1e-5
            )  # add epsilon to avoid 1.0 weight

        # Combine with weights computed from LF accuracies
        w = np.log(weights / (1 - weights))
        w[np.abs(w) == np.inf] = 0  # set weights from acc==0 to 0

        # TODO: add multiclass support
        L_np[L_np == 2] = -1
        label_probs = expit(2 * L_np @ w).reshape(-1, 1)
        Y_weak = np.concatenate((label_probs, 1 - label_probs), axis=1)
        return Y_weak
