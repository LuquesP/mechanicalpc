import os
import json
import numpy as np
from data_utils.pointsampling import PointSampling


class DataProcessor:
    def __init__(self, dataset_directory_path, filetype, points, dataset_type) -> None:
        self.dataset_directory_path = (
            os.path.join(dataset_directory_path, "train")
            if dataset_type == "train"
            else os.path.join(dataset_directory_path, "test")
        )
        self.points = points
        self.filetype = filetype
        self.dataset_type = dataset_type
        self.point_sampling = PointSampling(points, filetype)
        self.outpath = os.path.join(os.getcwd(), "temp")

    def save_pointcloud(self, pointcloud, filename):
        if not os.path.exists(self.outpath):
            os.mkdir(self.outpath)
        np.save(os.path.join(self.outpath, filename + ".npy"), pointcloud)

    def save_annotations(self, annotations):
        with open(
            os.path.join(self.outpath, "annot_" + self.dataset_type + ".json"), "w"
        ) as f:
            json.dump(annotations, f)

    def preprocess_dataset(self):
        annotations = {}
        cnt = 0
        for subdir, _, files in os.walk(self.dataset_directory_path):
            class_name = os.path.basename(subdir)
            for file in files:
                pointcloud = self.point_sampling.convert_to_pointcloud(
                    os.path.join(self.dataset_directory_path, subdir, file)
                )
                self.save_pointcloud(pointcloud, file.split(".")[0])
                annotations[cnt] = {
                    "annotation": class_name,
                    "filename": file.split(".")[0],
                }
                cnt += 1
        self.save_annotations(annotations)
        return annotations
