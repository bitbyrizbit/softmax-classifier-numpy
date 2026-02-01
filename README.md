# **Softmax Classifier From Scratch (NumPy)**

This project implements a **multi-class classification model from scratch using NumPy**, focusing on the **Softmax classifier trained with gradient descent**.
The goal is to understand how probabilistic multi-class models work internally, without relying on high-level machine learning libraries.

The implementation emphasizes **numerical stability, clean mathematical formulation, and transparent training logic**, making it suitable for learning, revision, and portfolio demonstration.

---

## **Objectives**

* Implement a **Softmax classifier from first principles**
* Understand **probability distributions over multiple classes**
* Train the model using **gradient descent**
* Evaluate performance using standard classification metrics
* Compare **Softmax vs One-vs-Rest (OvR)** approaches conceptually

---

## **Dataset**

The project uses the **Iris dataset**, a classic multi-class classification dataset with **three target classes**.
It is loaded directly from `sklearn.datasets` and used only as a data source - **no sklearn models are used**.

**Why Iris?**

* Clean, well-balanced, multi-class dataset
* Suitable for understanding decision boundaries
* Widely accepted for foundational ML implementations

---

## **Implementation Details**

### Core Components Implemented

* **Feature normalization** using NumPy
* **One-hot encoding** of class labels
* **Numerically stable Softmax function**
* **Categorical cross-entropy loss**
* **Analytical gradient computation**
* **Gradient descent training loop**

All computations are **fully vectorized** - no loops over samples.

---

## **Training Workflow**

1. Load and normalize the dataset
2. One-hot encode target labels
3. Initialize weights and bias
4. Forward pass (Linear -> Softmax)
5. Compute categorical cross-entropy loss
6. Backpropagate gradients
7. Update parameters using gradient descent
8. Track loss across epochs

---

## **Evaluation**

The trained model is evaluated using:

* **Accuracy**
* **Confusion Matrix**

These metrics help assess both overall performance and class-wise behavior.

---

## **Softmax vs One-vs-Rest (OvR)**

### Softmax

* Produces a **single probability distribution** over all classes
* Probabilities sum to 1
* Better calibrated confidence estimates

### One-vs-Rest (OvR)

* Trains one binary classifier per class
* Probabilities are independent
* Can lead to ambiguous confidence interpretation

**Softmax is preferred for true multi-class problems**, as implemented here.

---

## **Probability Interpretation**

Each output value represents the model’s **confidence that a sample belongs to a specific class**, relative to all others.
This makes Softmax especially useful when **class competition matters**, unlike independent binary classifiers.

---

## **Project Structure**

```
softmax-classifier-numpy/
│
├── main.py              # Complete Softmax classifier implementation
├── requirements.txt     # Minimal dependencies
├── .gitignore           # Git ignore rules
└── README.md            # Project documentation
```

---

## **How to Run**

1. Clone the repository:
   ```bash
   git clone https://github.com/bitbyrizbit/softmax-classifier-numpy.git

2. Navigate to the project directory:
   ```bash
   cd softmax-classifier-numpy

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Run the model:

   ```bash
   python main.py
   ```

The script trains the classifier and prints evaluation metrics.

---

## **Key Takeaways**

* Softmax provides a principled way to handle **multi-class probability modeling**
* Vectorization is critical for clean and efficient numerical code
* Implementing models from scratch builds **true algorithmic understanding**
