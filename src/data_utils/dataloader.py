import os
import numpy as np
import math
from torch.utils.data import Dataset
import torch
from torchvision import transforms, utils
import json

# from augmentations import Normalize, ToTensor, RandRotation_z, RandomNoise


class Datasetloader(Dataset):
    def __init__(self, root_dir, annotations, classes, transforms) -> None:
        self.root_dir = root_dir
        self.annotations = annotations
        self.classes = classes
        self.transforms = transforms

    def __len__(self):
        return len(self.annotations)

    def __preproc__(self, pc_path):
        pc = np.load(pc_path)
        pc = self.transforms(pc)
        return pc

    def __getitem__(self, idx):
        pc_path = os.path.join(
            self.root_dir, self.annotations[str(idx)]["filename"] + ".npy"
        )
        pc = self.__preproc__(pc_path)
        return {
            "pointcloud": pc,
            "category": torch.from_numpy(
                np.array(self.classes[self.annotations[str(idx)]["annotation"]]).astype(
                    np.int64
                )
            ),
        }
