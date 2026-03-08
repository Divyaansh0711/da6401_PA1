import numpy as np


def to_onehot(y, num_classes):
    """
    Convert labels to one-hot encoding if they are not already.
    Accepts:
        - 1D integer array shape (N,)  → converts to (N, num_classes)
        - 2D float array shape (N, C)  → returned as-is (already one-hot)
    """
    if y.ndim == 1:
        return np.eye(num_classes)[y.astype(int)]
    return y  # already one-hot


class CrossEntropyLoss:
    """
    Cross-Entropy loss with numerically stable softmax applied internally.

    The model outputs raw logits; softmax is applied here, NOT in the network.
    Accepts y_true as either:
        - integer labels,  shape (N,)    e.g. [0, 3, 9, ...]
        - one-hot labels,  shape (N, C)  e.g. [[1,0,...], ...]
    """

    def __init__(self):
        self.y_true = None   # stored as one-hot (N, C) after forward()
        self.y_pred = None   # softmax probabilities (N, C)
        self.num_classes = None

    def forward(self, logits, y_true):
        """
        Args:
            logits:  raw network output, shape (N, C)
            y_true:  integer labels (N,) OR one-hot labels (N, C)
        Returns:
            scalar cross-entropy loss
        """
        self.num_classes = logits.shape[1]
        # Convert to one-hot internally regardless of input format
        self.y_true = to_onehot(y_true, self.num_classes)

        # Numerically stable softmax
        shifted = logits - np.max(logits, axis=1, keepdims=True)
        exp = np.exp(shifted)
        self.y_pred = exp / np.sum(exp, axis=1, keepdims=True)

        # Cross-entropy: -sum(y_true * log(softmax)) / N
        loss = -np.sum(self.y_true * np.log(self.y_pred + 1e-12)) / self.y_true.shape[0]
        return loss

    def backward(self):
        """
        Combined gradient of softmax + cross-entropy w.r.t. logits.
        dL/d_logits = (softmax - y_true) / N
        """
        return (self.y_pred - self.y_true) / self.y_true.shape[0]


class MSELoss:
    """
    Mean Squared Error loss.
    No softmax — raw logits are compared directly to y_true.

    Convention: loss = sum((y_pred - y_true)^2) / N
    (sum over class dimension, mean over batch dimension)

    Accepts y_true as either integer labels (N,) or one-hot labels (N, C).
    """

    def __init__(self):
        self.y_true = None
        self.y_pred = None
        self.num_classes = None

    def forward(self, y_pred, y_true):
        """
        Args:
            y_pred:  raw logits, shape (N, C)
            y_true:  integer labels (N,) OR one-hot labels (N, C)
        Returns:
            scalar MSE loss = sum((y_pred - y_true_onehot)^2) / N
        """
        self.num_classes = y_pred.shape[1]
        # Convert to one-hot internally
        self.y_true = to_onehot(y_true, self.num_classes)
        self.y_pred = y_pred
        N = self.y_true.shape[0]
        # sum over classes, mean over batch
        loss = np.sum((self.y_true - self.y_pred) ** 2) / N
        return loss

    def backward(self):
        """
        dL/d_y_pred = 2*(y_pred - y_true) / N
        (consistent with forward: sum/N convention)
        """
        N = self.y_true.shape[0]
        return 2.0 * (self.y_pred - self.y_true) / N