import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.metrics import confusion_matrix

from src.utils import normalize_features, one_hot_encode
from src.model import SoftmaxClassifier
from experiments.decision_boundary import plot_decision_boundary

# Step 1: Load dataset
data = load_iris()
X = data.data[:, :2]        # only 2 features for visualization
y = data.target

num_classes = len(np.unique(y))

# Step 2: Normalize features
X = normalize_features(X)

# Step 3: One-hot encode labels
y_onehot = one_hot_encode(y, num_classes)

# Step 4: Initialize model
model = SoftmaxClassifier(
    input_dim=X.shape[1],
    num_classes=num_classes,
    lr=0.1
)

# Step 5: Train model using gradient descent
model.train(X, y_onehot, epochs=1000)

# Step 10: Evaluation
y_pred = model.predict(X)
accuracy = np.mean(y_pred == y)

print("Accuracy:", accuracy)
print("Confusion Matrix:\n", confusion_matrix(y, y_pred))

# Step 11: Visualization
plt.plot(model.loss_history)
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training Loss Curve")
plt.show()

plot_decision_boundary(model, X, y)
