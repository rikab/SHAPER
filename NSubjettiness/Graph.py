import torch
import torch.nn
import torch.nn.functional as F
import numpy as np

from matplotlib.patches import Polygon, Circle
from matplotlib.collections import PatchCollection


class graph():

    def __init__(self, structure, R, R_prime = np.inf):


        self.structure = structure
        self.R = R
        self.R_prime = R_prime

        num_anchors = 1
        min_index = 1
        for polygon in self.structure:
            if max(polygon) > num_anchors:
                num_anchors = max(polygon)
            if min(polygon) < min_index:
                min_index = min(polygon)

        self.num_anchors = num_anchors + 1 - min_index
        for i in range(len(self.structure)):
            self.structure[i] = np.array(self.structure[i]) - min_index





    def draw_polygons(self, ax, A):

        # Draw purple polygons
        patches = []
        for polygon in self.structure:
            if polygon.shape[0] == 1:
                patches.append(Circle(A[polygon[0]], self.R / 8, facecolor = 'purple'))
            else:
                patches.append(Polygon(A[polygon], True))
        p = PatchCollection(patches, color = "purple", alpha = 0.25)
        ax.add_collection(p)





