import numpy as np

class NeuralLayer:

    def __init__(self, input_dim, output_dim, weight_init="random"):
        self.input_dim = input_dim
        self.output_dim = output_dim
        if weight_init == "zeros":
            self.W = np.zeros((input_dim, output_dim))
            self.b = np.zeros((1, output_dim))
        elif weight_init == "random":
            self.W = np.random.randn(input_dim, output_dim) * 0.1
            self.b = np.zeros((1, output_dim))
        elif weight_init == "xavier":
            limit = np.sqrt(6/(input_dim+output_dim))
            self.W  = np.random.uniform(-limit, limit, (input_dim, output_dim))
            self.b = np.zeros((1, output_dim))
        else:
            raise ValueError("Invalid weight initialization method")
        # self.b = np.zeros((1, output_dim))

        self.grad_W = np.zeros((input_dim, output_dim))
        self.grad_b = np.zeros((1, output_dim))
        self.input = None

    def forward(self, x):
        # implementing z = wx + b
        # x is input self.input
        self.input = x
        z = np.dot(x, self.W) + self.b
        return z

    def backward(self, dz):

        self.grad_W = np.dot(self.input.T, dz)
        self.grad_b = np.sum(dz, axis=0, keepdims=True)

        dx = np.dot(dz, self.W.T)

        return dx
