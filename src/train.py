import os
import matplotlib.pyplot as plt

from .preprocess import get_train_validation_generators
from .model import build_model
from .config import MODEL_PATH, EPOCHS


def train():

    print("Preparing dataset...")

    train_gen, val_gen = get_train_validation_generators()

    print("Building CNN model...")

    model = build_model()

    print("Training started...")

    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS
    )

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

    model.save(MODEL_PATH)

    print(f"Model saved at {MODEL_PATH}")

    plot_training(history)


def plot_training(history):

    os.makedirs("outputs/plots", exist_ok=True)

    plt.figure()

    plt.plot(history.history["accuracy"])
    plt.plot(history.history["val_accuracy"])

    plt.title("Training vs Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")

    plt.legend(["Train", "Validation"])

    plt.savefig("outputs/plots/accuracy.png")

    print("Plot saved")


if __name__ == "__main__":
    train()