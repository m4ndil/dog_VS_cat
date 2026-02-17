import os
import pandas as pd
from tensorflow.keras.models import load_model

from .preprocess import get_test_generator
from .config import MODEL_PATH, PREDICTION_PATH


def predict():

    print("Loading model...")

    model = load_model(MODEL_PATH)

    test_gen = get_test_generator()

    print("Predicting...")

    predictions = model.predict(test_gen)

    labels = (predictions > 0.5).astype(int)

    df = pd.DataFrame({
        "filename": test_gen.filenames,
        "label": labels.flatten()
    })

    os.makedirs(os.path.dirname(PREDICTION_PATH), exist_ok=True)

    df.to_csv(PREDICTION_PATH, index=False)

    print(f"Predictions saved at {PREDICTION_PATH}")


if __name__ == "__main__":
    predict()