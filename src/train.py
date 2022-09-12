from difflib import get_close_matches
import os
import json
from utils.utils import get_classes
from utils.transforms import Normalize, ToTensor, RandRotation_z, RandomNoise
from torchvision import transforms, utils
from models.pointnet import PointNetCls, feature_transform_regularizer
from dataloader import MechanicalData


def default_transforms():
    return transforms.Compose([Normalize(), ToTensor()])


def train_transforms():
    return transforms.Compose(
        [Normalize(), RandRotation_z(), RandomNoise(), ToTensor()]
    )


dataset_path = r"C:\Users\LukasPilsl\source\studium\bachelor\mechanicalpc\data"
train_annot = {}
with open(os.path.join(dataset_path, "annot_train.json")) as f:
    train_annot = json.load(f)
classes = get_classes(train_annot)


train_ds = MechanicalData(dataset_path, train_annot, classes, train_transforms())

print(train_ds[0])
