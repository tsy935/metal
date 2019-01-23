import numpy as np
from snorkel.learning.gen_learning import GenerativeModel

from metal.label_model import LabelModel
from metal.utils import recursive_merge_dicts


class SnorkelLabelModel(LabelModel):
    """A wrapper that gives the Snorkel generative model a MeTaL interface"""

    def __init__(self):
        super().__init__()
        self.model = GenerativeModel()

    @staticmethod
    def categorical_to_plusminus(L):
        L = L.copy()
        L[L == 2] = -1
        return L

    def train_model(self, L, **kwargs):
        L = self.categorical_to_plusminus(L)
        self.model.train(L, verbose=False, **kwargs)

    def predict_proba(self, L):
        L = self.categorical_to_plusminus(L)
        marginals = self.model.marginals(L).reshape(-1, 1)
        Y_ps = np.hstack((marginals, 1 - marginals))
        return Y_ps
