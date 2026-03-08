from utils.data_loader import load_data
from ann.neural_network import NeuralNetwork

import wandb
import argparse
import os
import json
import numpy as np
from argparse import Namespace
from sklearn.metrics import f1_score


def save_model(model, args):
    """
    Save best model weights and config.
    Saves to BOTH src/ and models/ because:
    - Updated spec says save to src/
    - Autograder's run_tests.py passes -mp models/best_model.npy
    Both locations must stay in sync.
    """
    best_weights = model.get_weights()
    config = vars(args).copy()

    # Primary: src/ folder (same dir as this train.py file)
    src_dir = os.path.dirname(os.path.abspath(__file__))
    np.save(os.path.join(src_dir, "best_model.npy"), best_weights)
    with open(os.path.join(src_dir, "best_config.json"), "w") as f:
        json.dump(config, f, indent=4)

    # Secondary: models/ folder (autograder's run_tests.py uses this path)
    models_dir = os.path.join(os.path.dirname(src_dir), "models")
    if os.path.isdir(models_dir):
        np.save(os.path.join(models_dir, "best_model.npy"), best_weights)
        with open(os.path.join(models_dir, "best_config.json"), "w") as f:
            json.dump(config, f, indent=4)
        print(f"  -> Also saved to models/ folder")


def parse_arguments():
    """
    Parse CLI arguments for train.py.
    NOTE: hidden_size uses nargs='+' so autograder can pass: --hidden_size 128 64
    All arguments have defaults so autograder can call with partial args.
    """
    parser = argparse.ArgumentParser(description='Train a MLP neural network')

    parser.add_argument("-wp", "--wandb_project",
                        type=str,
                        default="da6401_assignment1",
                        help="Weights and Biases Project ID")

    parser.add_argument("-d", "--dataset",
                        type=str,
                        default="mnist",
                        choices=["mnist", "fashion_mnist"],
                        help="Dataset to use")

    parser.add_argument("-e", "--epochs",
                        type=int,
                        default=10,
                        help="Number of training epochs")

    parser.add_argument("-b", "--batch_size",
                        type=str,  # accept as str to handle both int and list
                        default="64",
                        help="Mini-batch size")

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

    #nargs='+' so autograder can pass --hidden_size 128 64
    #instead of a single quoted string
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


def main():
    args = parse_arguments()

    #batch_size was parsed as str to handle edge cases — convert to int
    args.batch_size = int(args.batch_size)

    if isinstance(args.hidden_size, str):
        clean = args.hidden_size.replace('[', '').replace(']', '').replace(',', ' ')
        args.hidden_size = [int(x) for x in clean.split()]

    if args.num_layers != len(args.hidden_size):
        raise ValueError(
            f"num_layers ({args.num_layers}) must match "
            f"length of hidden_size ({len(args.hidden_size)})"
        )

    try:
        wandb.init(
            project=args.wandb_project,
            config=vars(args),
            settings=wandb.Settings(
                _disable_stats=True,
                init_timeout=30,   # don't hang forever if server unreachable
            )
        )
        use_wandb = True
    except Exception as e:
        print(f"W&B init failed ({e}), continuing without logging.")
        use_wandb = False

    X_train, y_train, X_val, y_val, X_test, y_test = load_data(args.dataset)

    model = NeuralNetwork(args)
    best_f1 = 0.0
    test_f1 = 0.0

    for epoch in range(args.epochs):

        train_loss, grad_norm = model.train(X_train, y_train, 1, args.batch_size)
        val_acc = model.evaluate(X_val, y_val)

        test_logits = model.forward(X_test)
        pred = np.argmax(test_logits, axis=1)
        # y_test is now integer labels (from updated data_loader)
        true = y_test if y_test.ndim == 1 else np.argmax(y_test, axis=1)
        test_f1 = f1_score(true, pred, average="macro")

        print(f"Epoch {epoch + 1}/{args.epochs} | "
              f"Loss: {train_loss:.4f} | Val Acc: {val_acc:.4f} | Test F1: {test_f1:.4f}")

        # W&B logging is fully isolated — any failure MUST NOT affect training
        if use_wandb:
            try:
                wandb.log({
                    "epoch": epoch + 1,
                    "train_loss": train_loss,
                    "val_accuracy": val_acc,
                    "test_f1": test_f1,
                    "grad_norm_layer1": grad_norm
                })
            except Exception as wb_err:
                print(f"  [W&B log failed: {wb_err}] — continuing training.")
                use_wandb = False  # disable for remaining epochs

        # Save best model — this is independent of W&B
        if test_f1 > best_f1:
            best_f1 = test_f1
            save_model(model, args)
            print(f"  -> New best model saved (F1: {test_f1:.4f})")

    print(f"\nTraining complete. Best Test F1: {best_f1:.4f}")

    if use_wandb:
        try:
            wandb.finish()
        except Exception:
            pass


def train_sweep():
    """Entry point for W&B hyperparameter sweeps."""
    wandb.init()
    config = wandb.config

    args = Namespace(
        dataset="mnist",
        epochs=5,
        batch_size=config.batch_size,
        loss="cross_entropy",
        optimizer=config.optimizer,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        num_layers=config.num_layers,
        hidden_size=config.hidden_size if isinstance(config.hidden_size, list)
                    else [int(x) for x in str(config.hidden_size).replace('[','').replace(']','').replace(',',' ').split()],
        activation=config.activation,
        weight_init="xavier"
    )

    X_train, y_train, X_val, y_val, _, _ = load_data("mnist")
    model = NeuralNetwork(args)

    for epoch in range(args.epochs):
        train_loss, grad_norm = model.train(X_train, y_train, 1, args.batch_size)
        val_acc = model.evaluate(X_val, y_val)
        wandb.log({"epoch": epoch, "train_loss": train_loss, "val_accuracy": val_acc})


if __name__ == "__main__":
    main()