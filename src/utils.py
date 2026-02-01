import numpy as np

def normalize_features(X):
    """
    Standardizes features to zero mean and unit variance.

    WHY:
    - Softmax is sensitive to feature scale
    - Normalization stabilizes gradients
    - Accelerates convergence
    """
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    return (X - mean) / std


def one_hot_encode(y, num_classes):
    """
    Converts class labels into one-hot encoded vectors.

    Example:
    y = [0, 2, 1] -> 
    [[1,0,0],
     [0,0,1],
     [0,1,0]]
    """
    one_hot = np.zeros((y.shape[0], num_classes))
    one_hot[np.arange(y.shape[0]), y] = 1
    return one_hot
