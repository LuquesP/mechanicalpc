from __future__ import annotations
import os
import numpy as np
import json
from tesselation import Tesselation

dataset_path = r"C:\Users\LukasPilsl\source\studium\bachelor\dataset_org_norm"
source_path = r"C:\Users\LukasPilsl\source\studium\bachelor\mechanicalpc\data"
train_dataset = {}
test_dataset = {}


def transform_dataset(path, set):
    if set == "train":
        annot = train_dataset
    else:
        annot = test_dataset
    cnt = 0
    existing = os.listdir(os.path.join(source_path, "files"))
    existing = [e.split(".")[0] for e in existing]
    for folder in os.listdir(path):
        for obj in os.listdir(os.path.join(path, folder)):
            try:
                annot[cnt] = {"annotation": folder, "filename": obj.split(".")[0]}
                cnt += 1
                if obj.split(".")[0] not in existing:
                    tess = Tesselation(
                        os.path.join(path, folder, obj), format="obj", outputsize=2048
                    )
                    pc = tess.get_pointcloud()
                    print(f"tesselation of {obj}")
                    np.save(os.path.join(source_path, obj.split(".")[0] + ".npy"), pc)
            except:
                print("error" + obj)
    save_file(annot, set)


def filter_not_existing_files(annot):
    files = os.listdir(os.path.join(source_path, "files"))
    files = [f.split(".")[0] for f in files]
    for i in range(len(annot)):
        if annot[str(i)]["filename"] not in files:
            del annot[str(i)]
    return annot


def save_file(annot, set):
    with open(os.path.join(source_path, "annot_" + set + ".json"), "w") as f:
        json.dump(annot, f)


# transform_dataset(os.path.join(dataset_path, "train"), "train")
# transform_dataset(os.path.join(dataset_path, "test"), "test")
train_annot = {}
with open(os.path.join(source_path, "annot_train.json")) as f:
    train_annot = json.load(f)
train_annot = filter_not_existing_files(train_annot)
save_file(train_annot, "train")
test_annot = {}
with open(os.path.join(source_path, "annot_train.json")) as f:
    train_annot = json.load(f)
test_annot = filter_not_existing_files(test_annot)
save_file(test_annot, "test")
