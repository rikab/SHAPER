# Standard stuff
from abc import ABC, abstractmethod
from matplotlib.pyplot import axis
from matplotlib.patches import Circle as pltCircle
from matplotlib.patches import Polygon as pltPolygon
from matplotlib.collections import PatchCollection
import numpy as np
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tensorflow as tf


class Shape(ABC):

    @abstractmethod
    # A function to generate a list of points to approximate the shape
    def uniform_sample(self, N):
        pass

    @abstractmethod
    # Generate points according to the parameters
    def get_points(self,):
        pass

    def get_zs(self, ):
        return tf.ones(shape = (self.N,), dtype = tf.float64) * self.z / self.N

    @abstractmethod
    def draw(self, ax):
        pass

    @abstractmethod
    def __str__(self):
        pass



class Circle(Shape):

    def __init__(self, center, radius, z = 1.0, N = 100) -> None:

        # Circle parameters
        self.center = tf.Variable(center,
                             trainable = True,
                             dtype = tf.float64)
        self.radius = tf.Variable(radius,
                             trainable = True,
                             dtype = tf.float64)
        self.z = tf.Variable(z,
                             trainable = True,
                             dtype = tf.float64)
        self.parameters = [self.center, self.radius]
        self.N = N

        # Save a list of coordinates to approximate the shape
        self.t = self.uniform_sample(N)

    def uniform_sample(self, N):
        return 2 * np.pi * tf.keras.backend.random_uniform(shape=(N,), dtype=tf.float64)

    def get_points(self):
        xy = tf.stack([tf.math.cos(self.t), tf.math.sin(self.t)], axis = 1)
        return self.center + self.radius * xy


    def draw(self, ax):
        draw_circle = pltCircle(self.center, self.radius, facecolor = None, edgecolor = "purple",  fill = False,)
        ax.add_artist(draw_circle)

    def __str__(self):
        return "Radius: %.3f, z: %.3f GeV" % (self.radius.numpy(), self.z.numpy())



class Disk(Shape):

    def __init__(self, center, radius, z = 1.0, N = 100) -> None:

        # Circle parameters
        self.center = tf.Variable(center,
                             trainable = True,
                             dtype = tf.float64)
        self.radius = tf.Variable(radius,
                             trainable = True,
                             dtype = tf.float64)
        self.z = tf.Variable(z,
                             trainable = True,
                             dtype = tf.float64)
        self.parameters = [self.center, self.radius]
        self.N = N

        # Save a list of coordinates to approximate the shape
        self.t, self.r = self.uniform_sample(N)
        

    def uniform_sample(self, N):
        phi = 2 * np.pi * tf.keras.backend.random_uniform(shape=(N,), dtype=tf.float64)
        r =  np.sqrt(tf.keras.backend.random_uniform(shape=(N,), dtype=tf.float64))
        return phi, r

    def get_points(self):
        xy = self.r[:,None] * tf.stack([tf.math.cos(self.t), tf.math.sin(self.t)], axis = 1)
        return self.center + tf.math.abs(self.radius) * xy


    def draw(self, ax):
        draw_circle = [pltCircle(self.center, tf.math.abs(self.radius), facecolor = None, edgecolor = "purple", )]
        p = PatchCollection(draw_circle, color = "purple", alpha = 0.25)
        ax.add_collection(p)
    
    def __str__(self):
        return "Radius: %.3f, z: %.3f GeV" % (tf.math.abs(self.radius).numpy(), self.z.numpy())




class Background(Shape):

    def __init__(self, lower_left, upper_right, z = 1.0, N = 100):

        # Background parameters
        self.lower_left = tf.Variable(lower_left,
                                trainable = False,
                                dtype = tf.float64)
        
        self.upper_right = tf.Variable(upper_right,
                                trainable = False,
                                dtype = tf.float64)

        self.z = tf.Variable(z,
                             trainable = True,
                             dtype = tf.float64)
        self.parameters = []
        self.N = N

        # Save a list of coordinates to approximate the shape
        self.t = self.uniform_sample(N)

    def uniform_sample(self, N):
        return tf.keras.backend.random_uniform(shape=(N,2), dtype=tf.float64)

    def get_points(self):
        return self.lower_left + (self.upper_right - self.lower_left) * self.t


    def draw(self, ax):
        pass
            
    def __str__(self):
        return "Background: %.3f GeV" % (self.z.numpy())


class N_Particle_Event(Shape):

    def __init__(self, points, z = 1.0, N = 1):

        # N jet parameters
        self.points = tf.Variable(points,
                             trainable = True,
                             dtype = tf.float64)


        self.z = tf.Variable(z,
                             trainable = True,
                             dtype = tf.float64)

        self.parameters = [self.points]
        self.N = N

        # Save a list of coordinates to approximate the shape
        self.t = self.uniform_sample(N)

    def uniform_sample(self, N):
        return tf.keras.backend.random_uniform(shape=(N,1), dtype=tf.float64)

    def get_points(self):
        return self.points

    def draw(self, ax):
        ax.scatter(self.points[:,0], self.points[:,1], color = "purple")

    def __str__(self):
        return ""


class Graph():

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
        A = A.numpy()
        for polygon in self.structure:
            if polygon.shape[0] == 1:
                patches.append(Circle(A[polygon[0]], self.R / 8, facecolor = 'purple'))
            else:
                patches.append(pltPolygon(A[polygon], True))
        p = PatchCollection(patches, color = "purple", alpha = 0.25)
        ax.add_collection(p)



