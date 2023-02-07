import os
import numpy as np
import json
from torchvision import transforms
from utils.utils import get_classes
from data_utils.data_preprocess import DataProcessor
from data_utils.augmentations import ToTensor, Normalize, RandomNoise, RandRotation_z
from data_utils.dataloader import Datasetloader


class Dataset:
    def __init__(self, dataset_directory, filetype, points, dataset_type):
        self.dataset_type = dataset_type
        self.points = points
        self.data_processor = DataProcessor(
            dataset_directory, filetype, points, dataset_type
        )
        self.dataset_directory = dataset_directory
        # self.dataloader = Dataloader()

    def get_transformations(self, augmentations):
        tansformations = [Normalize(), ToTensor()]
        for augmentation in augmentations:
            if augmentation == "RandomNoise":
                tansformations += [RandomNoise()]
            if augmentation == "RandRotation_z":
                tansformations += [RandRotation_z()]
        return transforms.Compose(tansformations)

    def create_dataset(self, augmentations):
        # converts fiels in dataset and returns pytorch dataset
        annotations = self.data_processor.preprocess_dataset()
        classes = get_classes(annotations)
        data_loader = Datasetloader(
            os.path.join(os.getcwd(), "temp"), annotations, classes, augmentations
        )
        return annotations, data_loader
