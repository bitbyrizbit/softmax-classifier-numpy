import numpy as np

def categorical_cross_entropy(y_true, y_pred):
    """
    Computes mean categorical cross-entropy loss.

    WHY:
    - Aligns with probabilistic interpretation
    - Penalizes confident wrong predictions
    """
    eps = 1e-8  # prevents log(0)
    return -np.mean(np.sum(y_true * np.log(y_pred + eps), axis=1))
