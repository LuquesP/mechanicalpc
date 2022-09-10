from OCC.Extend.DataExchange import read_step_file
from OCC.Core.Tesselator import ShapeTesselator
import numpy as np
import random


class Tesselation:
    def __init__(self, filepath, outputsize=1024) -> None:
        self.filepath = filepath
        self.outputsize = outputsize
        self.pc = np.zeros((outputsize, 3))
        self.faces = []
        self.idxs = []

    def get_pointcloud(self):
        self.__tesselation()
        self.__pc_sampling()
        return self.pc

    def __tesselation(self):
        shape = read_step_file(self.filepath)
        tess = ShapeTesselator(shape)
        tess.Compute()
        for i in range(tess.ObjGetTriangleCount()):
            idx1, idx2, idx3 = tess.GetTriangleIndex(i)
            self.faces.append(
                [tess.GetVertex(idx1), tess.GetVertex(idx2), tess.GetVertex(idx3)]
            )
        self.idxs = [i for i in range(len(self.faces))]
        self.faces = np.array(self.faces)

    def __face_area(self, pt1, pt2, pt3):
        side_a = np.linalg.norm(pt1 - pt2)
        side_b = np.linalg.norm(pt2 - pt3)
        side_c = np.linalg.norm(pt3 - pt1)
        s = 0.5 * (side_a + side_b + side_c)
        return max(s * (s - side_a) * (s - side_b) * (s - side_c), 0) ** 0.5

    def __sample_point(self, pt1, pt2, pt3):
        # barycentric coordinates on a triangle
        # https://mathworld.wolfram.com/BarycentricCoordinates.html
        s, t = sorted([random.random(), random.random()])
        f = lambda i: s * pt1[i] + (t - s) * pt2[i] + (1 - t) * pt3[i]
        return (f(0), f(1), f(2))

    def __face_areas(self):
        face_areas = np.zeros(len(self.faces))
        for i in range(len(self.faces)):
            face_areas[i] = self.__face_area(
                self.faces[i][0], self.faces[i][1], self.faces[i][2]
            )
        return face_areas

    def __pc_sampling(self):
        a = self.__face_areas()
        sampled_faces = random.choices(
            self.idxs, weights=a, cum_weights=None, k=self.outputsize
        )
        for i, sampled_face in enumerate(sampled_faces):
            self.pc[i] = self.__sample_point(
                self.faces[sampled_face][0],
                self.faces[sampled_face][1],
                self.faces[sampled_face][2],
            )
