from imp import IMP_HOOK
import os
import numpy as np
import math
from torch.utils.data import Dataset
import torch
from torchvision import transforms, utils
import json
from utils.utils import get_classes
from utils.transforms import Normalize, ToTensor, RandRotation_z, RandomNoise


class MechanicalData(Dataset):
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
            self.root_dir, "files", self.annotations[str(idx)]["filename"] + ".npy"
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


def get_dataset(path, set):
    if set == "train":
        data_transforms = transforms.Compose(
            [Normalize(), RandRotation_z(), RandomNoise(), ToTensor()]
        )
    elif set == "test":
        data_transforms = transforms.Compose([Normalize(), ToTensor()])
    else:
        raise Exception(f"not supportet set: {set}")
    annotations = {}
    with open(os.path.join(path, f"annot_{set}.json")) as f:
        annotations = json.load(f)
    classes = get_classes(annotations)
    dataset = MechanicalData(path, annotations, classes, data_transforms)
    return annotations, dataset
