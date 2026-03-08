import numpy as np
from .neural_layer import NeuralLayer
from .activations import ReLU, Sigmoid, Tanh
from .objective_functions import CrossEntropyLoss, MSELoss, to_onehot
from .optimizers import SGD, Momentum, NAG, RMSProp


class NeuralNetwork:

    def __init__(self, cli_args):
        self.args = cli_args

        # Safely normalize all string args to avoid AttributeError on None
        self.args.optimizer   = str(self.args.optimizer).lower()   if self.args.optimizer   else "sgd"
        self.args.activation  = str(self.args.activation).lower()  if self.args.activation  else "relu"
        self.args.loss        = str(self.args.loss).lower()        if self.args.loss        else "cross_entropy"
        self.args.weight_init = str(self.args.weight_init).lower() if self.args.weight_init else "xavier"

        # Ensure hidden_size is a list of ints — handles string, int, or list input
        if isinstance(self.args.hidden_size, (int, float)):
            self.args.hidden_size = [int(self.args.hidden_size)]
        elif isinstance(self.args.hidden_size, str):
            clean = self.args.hidden_size.replace('[','').replace(']','').replace(',',' ')
            self.args.hidden_size = [int(x) for x in clean.split()]
        else:
            self.args.hidden_size = [int(x) for x in self.args.hidden_size]

        # Ensure weight_decay exists and is a float
        if not hasattr(self.args, 'weight_decay') or self.args.weight_decay is None:
            self.args.weight_decay = 0.0
        self.args.weight_decay = float(self.args.weight_decay)

        self.layers = []
        self.activations = []

        input_dim  = 784
        output_dim = 10
        hidden_sizes    = self.args.hidden_size
        activation_name = self.args.activation
        weight_init     = self.args.weight_init
        prev_dim = input_dim

        # ── BUILD HIDDEN LAYERS ──────────────────────────────────────────────
        for hidden_dim in hidden_sizes:
            self.layers.append(NeuralLayer(prev_dim, hidden_dim, weight_init))

            if activation_name == "relu":
                self.activations.append(ReLU())
            elif activation_name == "sigmoid":
                self.activations.append(Sigmoid())
            elif activation_name == "tanh":
                self.activations.append(Tanh())
            else:
                raise ValueError(f"Invalid activation: {activation_name}")

            prev_dim = hidden_dim

        #output layer

        self.layers.append(NeuralLayer(prev_dim, output_dim, weight_init))

        #loss fnc
        if self.args.loss == "cross_entropy":
            self.loss = CrossEntropyLoss()
        elif self.args.loss == "mse":
            self.loss = MSELoss()
        else:
            raise ValueError(f"Invalid loss: {self.args.loss}")

        #optimiser
        lr  = float(self.args.learning_rate)
        wd  = float(self.args.weight_decay)
        opt = self.args.optimizer

        if opt == "sgd":
            self.optimizer = SGD(self.layers, lr, weight_decay=wd)
        elif opt == "momentum":
            self.optimizer = Momentum(self.layers, lr, weight_decay=wd)
        elif opt == "nag":
            self.optimizer = NAG(self.layers, lr, weight_decay=wd)
        elif opt == "rmsprop":
            self.optimizer = RMSProp(self.layers, lr, weight_decay=wd)
        else:
            raise ValueError(f"Invalid optimizer: {opt}")

    def forward(self, X):
        a = X
        for layer, activation in zip(self.layers[:-1], self.activations):
            z = layer.forward(a)
            a = activation.forward(z)

        logits = self.layers[-1].forward(a)
        return logits

    def backward(self, y_true, y_pred):
        # Populate loss state — handles both integer and one-hot y_true
        self.loss.forward(y_pred, y_true)
        dz = self.loss.backward()

        # Output layer backward
        dz = self.layers[-1].backward(dz)

        # Hidden layers backward (reverse order)
        for layer, activation in reversed(list(zip(self.layers[:-1], self.activations))):
            dz = activation.backward(dz)
            dz = layer.backward(dz)

        # Return gradients from LAST layer → FIRST layer (per updated spec)
        grad_W = [layer.grad_W for layer in reversed(self.layers)]
        grad_b = [layer.grad_b for layer in reversed(self.layers)]

        return grad_W, grad_b

    def update_weights(self):
        self.optimizer.step()

    def train(self, X_train, y_train, epochs, batch_size):
        avg_loss = 0.0
        first_layer_grad_norm = 0.0

        for epoch in range(epochs):
            # Shuffle training data
            indices = np.random.permutation(X_train.shape[0])
            X_train = X_train[indices]
            y_train = y_train[indices]

            total_loss  = 0.0
            num_batches = 0

            for i in range(0, X_train.shape[0], batch_size):
                X_batch = X_train[i:i + batch_size]
                y_batch = y_train[i:i + batch_size]

                # Forward
                y_pred = self.forward(X_batch)

                # Loss (also stores state for backward)
                loss = self.loss.forward(y_pred, y_batch)
                total_loss  += loss
                num_batches += 1

                # Backward — grads stored in each layer
                self.backward(y_batch, y_pred)
                first_layer_grad_norm = np.linalg.norm(self.layers[0].grad_W)

                # Update weights (weight decay applied here)
                self.update_weights()

            avg_loss = total_loss / num_batches

        return avg_loss, first_layer_grad_norm

    def evaluate(self, X, y):
        y_pred = self.forward(X)
        predictions = np.argmax(y_pred, axis=1)

        # Handle both integer and one-hot y
        if y.ndim == 1:
            true_labels = y.astype(int)
        else:
            true_labels = np.argmax(y, axis=1)

        return np.mean(predictions == true_labels)

    def get_weights(self):
        weights = {}
        for i, layer in enumerate(self.layers):
            weights[f"W{i}"] = layer.W
            weights[f"b{i}"] = layer.b
        return weights

    def set_weights(self, weights):
        for i, layer in enumerate(self.layers):
            layer.W = weights[f"W{i}"]
            layer.b = weights[f"b{i}"]