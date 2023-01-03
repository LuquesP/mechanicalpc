# from OCC.Extend.DataExchange import read_step_file
# from OCC.Core.Tesselator import ShapeTesselator
import trimesh
import numpy as np
import random
import os
import json


class PointSampling:
    def __init__(self, points, filetype) -> None:
        self.points = points
        self.filetype = filetype

    def mesh_from_step(self, filepath):
        pass

    def mesh_from_obj(self, filepath):
        mesh = trimesh.load(filepath, force="mesh")
        # type: ignore
        return np.array(mesh.vertices), np.array(mesh.faces)

    def triangle_area(self, pt1, pt2, pt3):
        side_a = np.linalg.norm(pt1 - pt2)
        side_b = np.linalg.norm(pt2 - pt3)
        side_c = np.linalg.norm(pt3 - pt1)
        s = 0.5 * (side_a + side_b + side_c)
        return max(s * (s - side_a) * (s - side_b) * (s - side_c), 0) ** 0.5

    def point_sampling(self, pt1, pt2, pt3):
        # barycentric coordinates on a triangle
        # https://mathworld.wolfram.com/BarycentricCoordinates.html
        s, t = sorted([random.random(), random.random()])
        f = lambda i: s * pt1[i] + (t - s) * pt2[i] + (1 - t) * pt3[i]
        return (f(0), f(1), f(2))

    def convert_to_pointcloud(self, filepath):
        pointcloud = np.zeros((self.points, 3))
        verts, faces = [], []
        if self.filetype == "step":
            # verts, faces = self.mesh_from_step(filepath)
            pass
        elif self.filetype == "obj":
            verts, faces = self.mesh_from_obj(filepath)
        else:
            raise AttributeError(f"filetype: {self.filetype} only obj and step")
        triangle_points = [[verts[f[0]], verts[f[1]], verts[f[2]]] for f in faces]
        triangle_areas = [
            self.triangle_area(verts[f[0]], verts[f[1]], verts[f[2]]) for f in faces
        ]
        random_faces = random.choices(
            list(range(0, len(triangle_areas))), weights=triangle_areas, k=self.points
        )
        for i, random_face in enumerate(random_faces):
            pointcloud[i] = self.point_sampling(
                triangle_points[random_face][0],
                triangle_points[random_face][1],
                triangle_points[random_face][2],
            )

        return pointcloud


class DataProcessor:
    def __init__(self, dataset_directory_path, points, filetype, set) -> None:
        self.dataset_directory_path = (
            os.path.join(dataset_directory_path, "train")
            if set == "train"
            else os.path.join(dataset_directory_path, "test")
        )
        self.points = points
        self.filetype = filetype
        self.set = set
        self.point_sampling = PointSampling(points, filetype)
        self.outpath = os.path.join(os.getcwd(), "temp")

    def save_pointcloud(self, pointcloud, filename):
        if not os.path.exists(self.outpath):
            os.mkdir(self.outpath)
        np.save(os.path.join(self.outpath, filename + ".npy"), pointcloud)

    def save_annotations(self, annotations):
        with open(os.path.join(self.outpath, "annot_" + self.set + ".json"), "w") as f:
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
        self.save_annotations(annotations)
