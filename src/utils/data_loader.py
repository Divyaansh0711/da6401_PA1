from tensorflow.keras.datasets import mnist, fashion_mnist
import numpy as np
from sklearn.model_selection import train_test_split


def load_data(dataset_name="mnist"):
    if dataset_name == "mnist":
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
    elif dataset_name == "fashion_mnist":
        (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    else:
        raise ValueError(f"Invalid dataset name: {dataset_name}")

    # Convert to float32 and normalize to [0, 1]
    x_train = x_train.astype(np.float32) / 255.0
    x_test  = x_test.astype(np.float32)  / 255.0

    # Flatten from (N, 28, 28) to (N, 784)
    x_train = x_train.reshape(x_train.shape[0], -1)
    x_test  = x_test.reshape(x_test.shape[0], -1)

    # Keep labels as integers — NO one-hot encoding here.
    # The loss functions handle one-hot conversion internally.
    y_train = y_train.astype(np.int64)
    y_test  = y_test.astype(np.int64)

    # Train/validation split (90/10)
    x_train, x_val, y_train, y_val = train_test_split(
        x_train, y_train, test_size=0.1, random_state=42
    )

    return x_train, y_train, x_val, y_val, x_test, y_test