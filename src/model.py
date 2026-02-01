import numpy as np
from src.softmax import softmax
from src.loss import categorical_cross_entropy

class SoftmaxClassifier:
    def __init__(self, input_dim, num_classes, lr=0.1):
        """
        Initializes model parameters.

        Small random weights break symmetry.
        Bias initialized to zero.
        """
        self.W = 0.01 * np.random.randn(input_dim, num_classes)
        self.b = np.zeros((1, num_classes))
        self.lr = lr
        self.loss_history = []

    def forward(self, X):
        """
        Forward pass:
        Linear scores -> softmax probabilities
        """
        Z = X @ self.W + self.b
        return softmax(Z)

    def backward(self, X, y_true, y_pred):
        """
        Gradient computation.

        dZ = y_pred - y_true
        """
        m = X.shape[0]
        dZ = y_pred - y_true

        dW = (1 / m) * (X.T @ dZ)
        db = (1 / m) * np.sum(dZ, axis=0, keepdims=True)

        return dW, db

    def train(self, X, y, epochs=1000):
        """
        Full gradient descent training loop.
        """
        for _ in range(epochs):
            y_pred = self.forward(X)
            loss = categorical_cross_entropy(y, y_pred)
            self.loss_history.append(loss)

            dW, db = self.backward(X, y, y_pred)

            self.W -= self.lr * dW
            self.b -= self.lr * db

    def predict(self, X):
        """
        Converts probabilities to class predictions.
        """
        probs = self.forward(X)
        return np.argmax(probs, axis=1)
