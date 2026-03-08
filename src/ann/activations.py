import numpy as np

class ReLU:

    def __init__(self):
        self.input = None

    def forward(self, x):
        self.input = x
        return np.maximum(0, x)

    def backward(self, dz):
        # dz is output of the previous layer
        dx = dz * (self.input > 0)
        return dx

class Sigmoid:

    def __init__(self):
        self.output = None

    def forward(self, x):
        self.output = 1/(1+np.exp(-x))
        return self.output

    def backward(self, dz):
        return dz * self.output * (1 - self.output)

class Tanh:

    def __init__(self):
        self.output = None

    def forward(self, x):
        self.output = np.tanh(x)
        return self.output

    def backward(self, dz):
        return dz * (1-self.output ** 2)

class Softmax:

    def __init__(self):
        self.output = None

    def forward(self, x):
        exp = np.exp(x - np.max(x, axis=1, keepdims = True))
        self.output = exp/np.sum(exp, axis=1, keepdims = True)
        return self.output

    def backward(self, dz):
        return dz

# relu = ReLU()
# x = np.array([-1,2,-3])
# print(relu.forward(x))



