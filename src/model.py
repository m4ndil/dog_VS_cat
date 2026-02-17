from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from .config import IMG_SIZE, LEARNING_RATE


def build_model():

    model = Sequential()

    model.add(Conv2D(32, (3, 3), activation="relu",
                     input_shape=(IMG_SIZE, IMG_SIZE, 3)))
    model.add(MaxPooling2D(2, 2))

    model.add(Conv2D(64, (3, 3), activation="relu"))
    model.add(MaxPooling2D(2, 2))

    model.add(Conv2D(128, (3, 3), activation="relu"))
    model.add(MaxPooling2D(2, 2))

    model.add(Flatten())

    model.add(Dense(512, activation="relu"))
    model.add(Dropout(0.5))

    model.add(Dense(1, activation="sigmoid"))

    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return model