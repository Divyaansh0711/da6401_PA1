import numpy as np


class SGD:
    """
    Simple Stochastic Gradient Descent (mini-batch).
    W = W - lr * (grad_W + weight_decay * W)
    Weight decay (L2 regularization) is applied here in the update step,
    NOT in the backward pass, so gradient checks see pure loss gradients.
    """
    def __init__(self, layers, lr, weight_decay=0.0):
        self.layers = layers
        self.lr = lr
        self.weight_decay = weight_decay

    def step(self):
        for layer in self.layers:
            grad_W = layer.grad_W + self.weight_decay * layer.W
            grad_b = layer.grad_b
            layer.W -= self.lr * grad_W
            layer.b -= self.lr * grad_b


class Momentum:
    """
    SGD with Momentum.
    v = beta * v + lr * (grad_W + weight_decay * W)
    W = W - v
    """
    def __init__(self, layers, lr, beta=0.9, weight_decay=0.0):
        self.layers = layers
        self.lr = lr
        self.beta = beta
        self.weight_decay = weight_decay
        self.v_W = [np.zeros_like(layer.W) for layer in layers]
        self.v_b = [np.zeros_like(layer.b) for layer in layers]

    def step(self):
        for i, layer in enumerate(self.layers):
            grad_W = layer.grad_W + self.weight_decay * layer.W
            self.v_W[i] = self.beta * self.v_W[i] + self.lr * grad_W
            self.v_b[i] = self.beta * self.v_b[i] + self.lr * layer.grad_b
            layer.W -= self.v_W[i]
            layer.b -= self.v_b[i]


class NAG:
    """
    Nesterov Accelerated Gradient.
    """
    def __init__(self, layers, lr, beta=0.9, weight_decay=0.0):
        self.layers = layers
        self.lr = lr
        self.beta = beta
        self.weight_decay = weight_decay
        self.v_W = [np.zeros_like(layer.W) for layer in layers]
        self.v_b = [np.zeros_like(layer.b) for layer in layers]

    def step(self):
        for i, layer in enumerate(self.layers):
            grad_W = layer.grad_W + self.weight_decay * layer.W
            v_W_prev = self.v_W[i].copy()
            v_b_prev = self.v_b[i].copy()
            self.v_W[i] = self.beta * self.v_W[i] - self.lr * grad_W
            self.v_b[i] = self.beta * self.v_b[i] - self.lr * layer.grad_b
            layer.W += -self.beta * v_W_prev + (1 + self.beta) * self.v_W[i]
            layer.b += -self.beta * v_b_prev + (1 + self.beta) * self.v_b[i]


class RMSProp:
    """
    RMSProp optimizer.
    s = beta * s + (1 - beta) * grad_W^2
    W = W - lr * (grad_W + wd*W) / (sqrt(s) + eps)
    """
    def __init__(self, layers, lr, beta=0.9, epsilon=1e-8, weight_decay=0.0):
        self.layers = layers
        self.lr = lr
        self.beta = beta
        self.eps = epsilon
        self.weight_decay = weight_decay
        self.s_W = [np.zeros_like(layer.W) for layer in layers]
        self.s_b = [np.zeros_like(layer.b) for layer in layers]

    def step(self):
        for i, layer in enumerate(self.layers):
            grad_W = layer.grad_W + self.weight_decay * layer.W
            self.s_W[i] = self.beta * self.s_W[i] + (1 - self.beta) * (grad_W ** 2)
            self.s_b[i] = self.beta * self.s_b[i] + (1 - self.beta) * (layer.grad_b ** 2)
            layer.W -= self.lr * grad_W / (np.sqrt(self.s_W[i]) + self.eps)
            layer.b -= self.lr * layer.grad_b / (np.sqrt(self.s_b[i]) + self.eps)