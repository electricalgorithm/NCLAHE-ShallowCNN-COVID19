"""
This module contains the COVIDDatasetReader class which reads
the COVID-19 dataset and constructs a pandas.DataFrame to
process data further.
"""

import os
from PIL import Image
import pandas as pd
import numpy as np
import albumentations
import multiprocessing

from logger import logger

class COVIDDatasetReader:
    """
    The class which reads the COVID-19 dataset and constructs
    a pandas.DataFrame to process data further.
    """

    def __init__(
        self,
        dir_location: str = None,
        dir_attr: bool = None,
        preloaded_data: pd.DataFrame = None,
    ):
        if preloaded_data is None:
            # Check if the dir_location is None.
            if dir_location is None:
                raise ValueError("The dir_location parameter must be a string.")
            # Check if the dir_attr is None.
            if dir_attr is None:
                raise ValueError("The dir_attr parameter must be a string.")

        # Save the parameters.
        self.dir_masks = os.path.join(dir_location, "masks") if dir_location else None
        self.dir_images = os.path.join(dir_location, "images") if dir_location else None
        self.dir_attr = dir_attr

        # Read the data automatically.
        self.data = self._read() if preloaded_data is None else preloaded_data

    @property
    def features(self) -> np.ndarray:
        """
        It returns the features of the dataset.
        :return: numpy.ndarray which has the features of the dataset.
        """
        return self.data.iloc[:, 0].values

    @property
    def labels(self) -> np.ndarray:
        """
        It returns the labels of the dataset.
        :return: numpy.ndarray which has the labels of the dataset.
        """
        return self.data.iloc[:, 1].values

    def add_augmented_data(self, prob: float = 1):
        """
        It creates augmented data by following methods:
            - Random Fog with probability given.
            - Random Gaussian Blur with probability given.
            - Random Brightness with probability given.
            - Random Contrast with probability given.
            - Random Vertical Flip with probability given.
            - Random Horizontal Flip with probability given.
            - Random Rotation with probability given.
        :param augmentor: The augmentor which is used to create augmented data.
        :return: None
        """
        albumentations_list =  [
            # albumentations.RandomFog(p=prob),
            albumentations.GaussianBlur(p=prob),
            albumentations.RandomBrightnessContrast(p=prob),
            albumentations.VerticalFlip(p=prob),
            albumentations.HorizontalFlip(p=prob)
        ]

        # Create the multiprocessing.Queue instance.
        multiprocessing.set_start_method("spawn")
        mp_queue = multiprocessing.Queue()
        for augmentor in albumentations_list:
            logger.debug("Augmenting data with %s" % augmentor.__class__.__name__)
            augmented_data = self.data.copy()
            process = multiprocessing.Process(
                target=self._augment,
                args=(augmented_data, augmentor, mp_queue)
            )
            process.start()
            logger.debug("Augmented data creator process started.")

        # Get the augmented data.
        for _ in range(len(albumentations_list)):
            augmented_data = mp_queue.get()
            logger.debug("Augmented data is read successfully.")
            self.data = pd.concat([self.data, augmented_data])
            logger.debug("Augmented data is added successfully.")

    def _augment(self, data_list_to_augment, augmentor, mp_queue) -> None:
        """This function augments the image data with the given augmentor.
        :param data_list_to_augment: The image data which is used to augment.
        :param augmentor: The augmentor which is used to augment the image data.
        :param mp_queue: The multiprocessing.Queue instance which is used to
        return the augmented image data.
        :return: None
        """
        data_list_to_augment["image"] = data_list_to_augment["image"].apply(
            lambda image_data: augmentor(image=np.array(image_data))["image"]
        )
        mp_queue.put(data_list_to_augment)
        logger.debug(f"Augmented data creator process ended for {augmentor.__class__.__name__}.")

    def _read(self) -> pd.DataFrame:
        """
        This function reads the PNG files inside the dir_location
        and constructs DataFrame which has two columns. The first
        column is the PNG data, the second column is the attribute
        of the dataset.
        :return: pandas.DataFrame which has two columns.
        """
        logger.info("Reading the images and masks.")

        # Read the PNG files.
        images = [
            Image.open(os.path.join(self.dir_images, file))
            for file in os.listdir(self.dir_images)
        ]
        logger.debug("Images are read successfully.")

        masks = [
            Image.open(os.path.join(self.dir_masks, file))
            for file in os.listdir(self.dir_masks)
        ]
        logger.debug("Masks are read successfully.")

        # Construct the DataFrame.
        data = pd.DataFrame(
            {"image": images, "masks": masks, "covid_status": self.dir_attr}
        )
        logger.debug("DataFrame is constructed successfully.")

        return data

    def __add__(self, other):
        """
        It adds two COVIDDatasetReader instances together.
        :param other: A COVIDDatasetReader instance.
        :return: A COVIDDatasetReader instance which combines both COVIDDatasetReaders.
        """
        # Check if the other is a COVIDDatasetReader instance.
        if not isinstance(other, COVIDDatasetReader):
            raise ValueError(
                "The other parameter must be a COVIDDatasetReader instance."
            )

        # Combine the two COVIDDatasetReaders and return it.
        data = pd.concat([self.data, other.data])
        logger.debug("Combined two COVIDDatasetReaders into new one.")
        return COVIDDatasetReader(preloaded_data=data)

    def __len__(self):
        """
        It returns the length of the dataset.
        :return: The length of the dataset.
        """
        return len(self.data)


if __name__ == "__main__":
    logger.info("COVIDDatasetReader is running as main.")

    # Define the dataset locations.
    COVID_DATASET_LOC = "../Dataset/COVID/"
    NORMAL_DATASET_LOC = "../Dataset/Normal/"

    # Read the datasets for COVID-19 cases.
    covid_pos = COVIDDatasetReader(dir_location=COVID_DATASET_LOC, dir_attr=True)
    covid_neg = COVIDDatasetReader(dir_location=COVID_DATASET_LOC, dir_attr=False)

    # All the cases.
    covid_all = covid_pos + covid_neg
    print(len(covid_all))

    # Add augmented data.
    covid_all.add_augmented_data()
    print(len(covid_all))
