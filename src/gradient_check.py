import numpy as np
from argparse import Namespace
from utils.data_loader import load_data
from ann.neural_network import NeuralNetwork


def compute_loss(model, X, y):
    y_pred = model.forward(X)
    return model.loss.forward(y_pred, y)


def gradient_check():

    # Small config
    args = Namespace(
        hidden_size=[5],
        activation="relu",
        weight_init="xavier",
        loss="cross_entropy",
        optimizer="sgd",
        learning_rate=0.01,
        weight_decay=0.0
    )

    # Load small subset of data
    X_train, y_train, _, _, _, _ = load_data("mnist")
    X = X_train[:5]
    y = y_train[:5]

    model = NeuralNetwork(args)

    # Forward
    y_pred = model.forward(X)

    # Backward
    model.loss.forward(y_pred, y)
    model.backward(y, y_pred)

    epsilon = 1e-5

    for layer_idx, layer in enumerate(model.layers):

        print(f"\nChecking Layer {layer_idx}")

        # Check W
        analytical_grad = layer.grad_W
        numerical_grad = np.zeros_like(layer.W)

        for i in range(layer.W.shape[0]):
            for j in range(layer.W.shape[1]):

                original_value = layer.W[i, j]

                # W + epsilon
                layer.W[i, j] = original_value + epsilon
                loss_plus = compute_loss(model, X, y)

                # W - epsilon
                layer.W[i, j] = original_value - epsilon
                loss_minus = compute_loss(model, X, y)

                # restore
                layer.W[i, j] = original_value

                numerical_grad[i, j] = (loss_plus - loss_minus) / (2 * epsilon)

        # Compute relative error
        numerator = np.linalg.norm(analytical_grad - numerical_grad)
        denominator = np.linalg.norm(analytical_grad) + np.linalg.norm(numerical_grad)

        relative_error = numerator / denominator

        print("Relative Error (W):", relative_error)

        assert relative_error < 1e-7, "Gradient check failed!"

    print("\nGradient check passed!")


if __name__ == "__main__":
    gradient_check()