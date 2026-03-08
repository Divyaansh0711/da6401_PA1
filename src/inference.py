import os
import json
import numpy as np
from argparse import Namespace
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from utils.data_loader import load_data
from ann.neural_network import NeuralNetwork


def parse_arguments():
    """
    The autograder calls inference.parse_arguments() directly.
    Same CLI as train.py per updated spec, with best config as defaults.
    """
    import argparse

    parser = argparse.ArgumentParser(description='Run inference on a trained MLP')

    parser.add_argument("-wp", "--wandb_project",
                        type=str,
                        default="da6401_assignment1",
                        help="Weights and Biases Project ID")

    parser.add_argument("-d", "--dataset",
                        type=str,
                        default="mnist",
                        choices=["mnist", "fashion_mnist"],
                        help="Dataset to evaluate on")

    parser.add_argument("-e", "--epochs",
                        type=int,
                        default=10,
                        help="Number of training epochs")

    parser.add_argument("-b", "--batch_size",
                        type=int,
                        default=64,
                        help="Batch size for inference")

    parser.add_argument("-l", "--loss",
                        type=str,
                        default="cross_entropy",
                        choices=["cross_entropy", "mse"],
                        help="Loss Function")

    parser.add_argument("-o", "--optimizer",
                        type=str,
                        default="rmsprop",
                        choices=["sgd", "momentum", "nag", "rmsprop"],
                        help="Optimizer type")

    parser.add_argument("-lr", "--learning_rate",
                        type=float,
                        default=0.001,
                        help="Learning rate")

    parser.add_argument("-wd", "--weight_decay",
                        type=float,
                        default=0.0,
                        help="L2 regularization strength")

    parser.add_argument("-nhl", "--num_layers",
                        type=int,
                        default=3,
                        help="Number of hidden layers")

    parser.add_argument("-sz", "--hidden_size",
                        nargs='+',
                        type=int,
                        default=[128, 128, 128],
                        help="Hidden layer sizes e.g. --hidden_size 128 64")

    parser.add_argument("-a", "--activation",
                        type=str,
                        default="relu",
                        choices=["relu", "sigmoid", "tanh"],
                        help="Activation function")

    parser.add_argument("-w_i", "--weight_init",
                        type=str,
                        default="xavier",
                        choices=["random", "xavier"],
                        help="Weight initialization method")

    parser.add_argument("-mp", "--model_path",
                        type=str,
                        default=None,
                        help="Path to saved model weights (.npy file)")

    return parser.parse_args()


def load_model(model_path):
    """
    Load trained model weights from disk.
    Exact pattern from updated assignment instructions.
    """
    data = np.load(model_path, allow_pickle=True).item()
    return data


def get_config(model_path):
    """
    Locate best_config.json.
    Search order:
      1. Same directory as the model file (models/ or src/)
      2. src/ directory (same as inference.py)
      3. models/ directory (autograder default location)
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))  # src/
    repo_root   = os.path.dirname(script_dir)                 # project root
    models_dir  = os.path.join(repo_root, "models")           # models/

    candidates = []
    if model_path:
        candidates.append(os.path.join(os.path.dirname(os.path.abspath(model_path)), "best_config.json"))
    candidates.append(os.path.join(script_dir, "best_config.json"))
    candidates.append(os.path.join(models_dir, "best_config.json"))

    config_path = None
    for path in candidates:
        if os.path.exists(path):
            config_path = path
            break

    if config_path is None:
        raise FileNotFoundError(
            f"best_config.json not found. Searched:\n" + "\n".join(f"  {p}" for p in candidates)
        )

    with open(config_path, "r") as f:
        config = json.load(f)

    # Ensure hidden_size is a list of ints
    if isinstance(config.get("hidden_size"), str):
        raw = config["hidden_size"].replace("[", "").replace("]", "").replace(",", " ")
        config["hidden_size"] = [int(x) for x in raw.split()]
    elif isinstance(config.get("hidden_size"), (int, float)):
        config["hidden_size"] = [int(config["hidden_size"])]
    elif not isinstance(config.get("hidden_size"), list):
        config["hidden_size"] = list(config["hidden_size"])
    else:
        config["hidden_size"] = [int(x) for x in config["hidden_size"]]

    return config


def evaluate_model(model, X_test, y_test):
    """Run forward pass and compute Accuracy, Precision, Recall, F1.
    y_test can be integer labels (N,) or one-hot (N, C) — both handled.
    """
    # Model returns raw LOGITS (not softmax) per updated spec
    logits = model.forward(X_test)

    predictions = np.argmax(logits, axis=1)
    # Handle both integer and one-hot y_test
    true_labels = y_test.astype(int) if y_test.ndim == 1 else np.argmax(y_test, axis=1)

    accuracy  = accuracy_score(true_labels, predictions)
    precision = precision_score(true_labels, predictions, average="macro", zero_division=0)
    recall    = recall_score(true_labels, predictions, average="macro", zero_division=0)
    f1        = f1_score(true_labels, predictions, average="macro", zero_division=0)

    loss = model.loss.forward(logits, y_test)

    return {
        "logits":    logits,
        "loss":      loss,
        "accuracy":  accuracy,
        "precision": precision,
        "recall":    recall,
        "f1":        f1
    }


def main():
    args = parse_arguments()

    # Determine model path — search src/ then models/ if not specified
    if args.model_path is None or not os.path.exists(args.model_path):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        repo_root  = os.path.dirname(script_dir)
        candidates = [
            os.path.join(script_dir, "best_model.npy"),          # src/best_model.npy
            os.path.join(repo_root, "models", "best_model.npy"), # models/best_model.npy
        ]
        for path in candidates:
            if os.path.exists(path):
                args.model_path = path
                break
        else:
            raise FileNotFoundError(
                f"best_model.npy not found. Searched:\n" +
                "\n".join(f"  {p}" for p in candidates)
            )

    # Load config from saved best_config.json (searches same dir as model, then fallbacks)
    config = get_config(args.model_path)
    cli = Namespace(**config)

    # Build model and load weights (exact pattern from updated spec)
    model = NeuralNetwork(cli)
    weights = load_model(args.model_path)
    model.set_weights(weights)

    # Load test data using dataset from CLI (allows evaluating on different dataset)
    _, _, _, _, X_test, y_test = load_data(args.dataset)

    results = evaluate_model(model, X_test, y_test)

    print("Evaluation Results")
    print("------------------")
    print(f"Loss:      {results['loss']:.4f}")
    print(f"Accuracy:  {results['accuracy']:.4f}")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall:    {results['recall']:.4f}")
    print(f"F1 Score:  {results['f1']:.4f}")

    return results


if __name__ == '__main__':
    main()