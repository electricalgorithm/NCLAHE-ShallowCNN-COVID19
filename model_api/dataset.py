"""
A helper module to load dataset from directory.
"""

import os
import cv2
import numpy
import pandas
from sklearn.model_selection import train_test_split


class Dataset:
    """
    A helper class to load dataset from directory.
    """

    def __init__(self, covid_dir: str, non_covid_dir: str) -> None:
        self.covid_dir: str = covid_dir
        self.non_covid_dir: str = non_covid_dir

    def load(
        self,
        split_sizes: list = None,
        img_sizes: tuple = (128, 128),
        max_number: int = 1000,
    ) -> list:
        """It loads the dataset and returns the image and label
        splits as list.

        Parameters
        ----------
        split_sizes : list
            Percentages of splits. [train_perc, val_perc, test_perc]
        img_sizes : str
            Sizes of the image. (img_X, img_Y)
            Defaults to (128, 128)
        max_number : int
            Number of maximum loaded dataset for each class.
            Defaults to 1000.

        Returns
        -------
        list
            [(trainX, trainY), (valX, valY), (testX, testY)]
        """
        if split_sizes is None:
            split_sizes = [0.6, 0.2, 0.2]

        if split_sizes[0] + split_sizes[1] + split_sizes[2] != 1.00:
            raise ValueError("Split sizes must be summed to 1.")

        images = []
        covid_image_count = 0
        for per_file in os.listdir(self.covid_dir):
            image = cv2.imread(os.path.join(self.covid_dir, per_file))
            image = cv2.resize(image, img_sizes)
            images.append([image / 255.0, 1])
            covid_image_count += 1

            if covid_image_count >= max_number:
                break

        non_image_count = 0
        for per_file in os.listdir(self.non_covid_dir):
            if per_file.startswith("Lung_Opacity"):
                continue
            image = cv2.imread(os.path.join(self.non_covid_dir, per_file))
            image = cv2.resize(image, img_sizes)
            images.append([image / 255.0, 0])
            non_image_count += 1

            if non_image_count >= (max_number * 2):
                break

        print("COVID: ", covid_image_count, "NON_COVID: ", non_image_count)
        dataset = pandas.DataFrame(images, columns=["Image", "Label"])
        inputs = dataset.loc[:, "Image"].to_list()
        labels = dataset.loc[:, "Label"].to_list()

        # Split the images.
        first_split = split_sizes[1] + split_sizes[2]
        images_train, images_non_train, label_train, label_non_train = train_test_split(
            inputs, labels, test_size=first_split, random_state=42
        )
        second_split = split_sizes[2] / first_split
        images_val, images_test, label_val, label_test = train_test_split(
            images_non_train, label_non_train, test_size=second_split, random_state=121
        )

        x_train = numpy.asarray(images_train).astype(numpy.float32)
        x_val = numpy.asarray(images_val).astype(numpy.float32)
        x_test = numpy.asarray(images_test).astype(numpy.float32)
        y_train = numpy.asarray(label_train).astype(numpy.intc)
        y_val = numpy.asarray(label_val).astype(numpy.intc)
        y_test = numpy.asarray(label_test).astype(numpy.intc)

        print("Train size: ", len(x_train))
        print("Validation size: ", len(x_val))
        print("Test size: ", len(x_test))

        return [(x_train, y_train), (x_val, y_val), (x_test, y_test)]
