"""
Module to create easier models with Keras API.
"""

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.metrics import BinaryAccuracy
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

import matplotlib.pyplot as plt


class CNNModel:
    """
    Handles the Keras API.
    """

    def __init__(self, name: str, input_sizes=(128, 128)) -> None:
        self.name = name
        self.model = self.__construct_model(input_sizes)
        self.callbacks = self.__construct_callbacks()

        # Internal attributes
        self.training_history = None

    def train(
        self, train_dataset: tuple, validation_dataset: tuple, epoch: int, batch: int
    ) -> dict:
        """Trains the model with Adam optimizer.

        Parameters
        ----------
        train_dataset : tuple
            (trainX, trainY)
        validation_dataset : tuple
            (valX, valY)

        Returns
        -------
        dict
            history object
        """
        # Train the model.
        self.training_history = self.model.fit(
            x=train_dataset[0],
            y=train_dataset[1],
            batch_size=batch,
            epochs=epoch,
            callbacks=self.callbacks,
            shuffle=True,
            workers=10,
            validation_data=validation_dataset,
        )

        return self.training_history

    def test(self, test_dataset: tuple) -> dict:
        """It tests the model with given dataset and saves the figures
        into the given parameter locations.

        Parameters
        ----------
        test_dataset : tuple
            (testX, testY)

        Returns
        -------
        dict
            {"loss": float, "binary_accuracy": float}
        """
        result = self.model.evaluate(test_dataset)
        return dict(zip(self.model.metrics_names, result))

    def save_figures(self) -> None:
        """
        Saves the figures of training process.
        """
        plt.plot(self.training_history.history["binary_accuracy"])
        plt.plot(self.training_history.history["val_binary_accuracy"])
        plt.title(f"Model Accuracy on {self.name}")
        plt.ylabel("Binary Accuracy")
        plt.xlabel("Epoch")
        plt.legend(["Train", "Validation"], loc="upper left")
        plt.savefig("{self.name}_accu.png")

        plt.plot(self.training_history.history["loss"])
        plt.plot(self.training_history.history["val_loss"])
        plt.title(f"Model Loss Function on {self.name}")
        plt.ylabel("Loss")
        plt.xlabel("Epoch")
        plt.legend(["Train", "Validation"], loc="upper left")
        plt.savefig("{self.name}_loss.png")

    def __construct_callbacks(self) -> list:
        """Creates Checkpoint and EarlyStopping callbacks.

        Returns
        -------
        list
           Callbacks list
        """
        filepath = f"{self.name}"
        filepath += ".epoch{epoch:02d}-acc{val_binary_accuracy:.2f}.hdf5"

        model_checkpoint_callback = ModelCheckpoint(
            filepath=filepath,
            save_weights_only=False,
            monitor="val_binary_accuracy",
            mode="max",
            save_best_only=True,
            verbose=1,
        )

        early_stopping = EarlyStopping(
            monitor="val_binary_accuracy", patience=5, min_delta=0.001, mode="max"
        )

        return [model_checkpoint_callback, early_stopping]

    def __construct_model(self, input_sizes: tuple) -> Sequential:
        """Creates the CNN model described in the article.

        Parameters
        ----------
        input_sizes : tuple
            The input sizes for the CNN model.
        """
        # Create the model.
        model = Sequential()

        # CNN layers
        model.add(
            Conv2D(
                input_shape=(input_sizes[0], input_sizes[1], 3),
                filters=32,
                kernel_size=(3, 3),
                activation="relu",
            )
        )
        model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(Dropout(0.1))
        model.add(
            Conv2D(
                filters=64,
                kernel_size=(3, 3),
                activation="relu",
            )
        )
        model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(Dropout(0.1))

        model.add(
            Conv2D(
                filters=128,
                kernel_size=(3, 3),
                activation="relu",
            )
        )
        model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

        model.add(Flatten())

        # Dense layers
        model.add(Dense(128, activation="relu"))
        model.add(Dense(10, activation="relu"))
        model.add(Dense(1, activation="sigmoid"))

        # Compile the model
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss=BinaryCrossentropy(),
            metrics=[BinaryAccuracy()],
        )

        return model
