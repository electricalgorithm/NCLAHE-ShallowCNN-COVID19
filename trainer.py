from model_api.dataset import Dataset
from model_api.cnn_model import CNNModel
from train_settings import settings


# Program itself.
if __name__ == "__main__":
    # Load the configs.
    SIZES = settings["MODEL_INPUT_SIZES"]
    COVID_DIR = settings["DIR_DATASET"] + settings["DIR_COVID"]
    NON_COVID_DIR = settings["DIR_DATASET"] + settings["DIR_NON_COVID"]
    MODEL_NAME = settings["MODEL_NAME"] + f"_{SIZES[0]}_{SIZES[1]}"
    EPOCH = settings["MODEL_EPOCH"]
    BATCH = settings["MODEL_BATCH"]

    # Load the dataset.
    dataset = Dataset(COVID_DIR, NON_COVID_DIR)
    [(trainX, trainY), (valX, valY), (testX, testY)] = dataset.load(img_sizes=SIZES)

    # Create and train the CNN model.
    model = CNNModel(MODEL_NAME, input_sizes=SIZES)
    model.train((trainX, trainY), (valX, valY), epoch=EPOCH, batch=BATCH)
    model.save_figures()

    # Test the model.
    results = model.test((testX, testY))
    print("Test Accuracy: ", results["accuracy"])
