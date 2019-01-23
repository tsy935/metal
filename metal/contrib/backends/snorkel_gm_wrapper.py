import numpy as np
from scipy.sparse import csr_matrix, issparse
from snorkel.learning.gen_learning import GenerativeModel

from metal.label_model import LabelModel


class SnorkelLabelModel(LabelModel):
    """A wrapper that gives the Snorkel generative model a MeTaL interface"""

    def __init__(self):
        super().__init__()
        self.model = GenerativeModel()

    def train_model(self, L, **kwargs):
        L = L.copy()
        if not issparse(L):
            L = csr_matrix(L)
        L[L == 2] = -1
        self.model.train(L, verbose=False, **kwargs)

    def predict_proba(self, L):
        L = L.copy()
        if not issparse(L):
            L = csr_matrix(L)
        L[L == 2] = -1
        marginals = self.model.marginals(L).reshape(-1, 1)
        Y_ps = np.hstack((marginals, 1 - marginals))
        return Y_ps
