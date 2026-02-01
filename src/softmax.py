import numpy as np

def softmax(Z):
    """
    Computes softmax probabilities row-wise.

    Numerical stability:
    Subtracting max prevents exp overflow.
    """
    Z_stable = Z - np.max(Z, axis=1, keepdims=True)
    exp_Z = np.exp(Z_stable)
    return exp_Z / np.sum(exp_Z, axis=1, keepdims=True)
