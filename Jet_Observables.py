

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from src.Observables import CustomObservable, Observable
import numpy as np
import tensorflow as tf
from pyjet import cluster


from matplotlib.patches import Circle as pltCircle
from matplotlib.patches import Ellipse as pltEllipse
from matplotlib.patches import Polygon as pltPolygon
from matplotlib.collections import PatchCollection

class Point(Observable):

    def __init__(self, initializer = None, z = 1.0, R = 0.5, beta = 1.0, ):
        super().__init__()

        # N jet parameters
        self.initializer = initializer
        if initializer != "kt":
            self.points = tf.Variable(initializer,
                                trainable = True,
                                dtype = tf.float64)
            self.parameters = [self.points]


        self.z = tf.Variable(z,
                             trainable = True,
                             dtype = tf.float64)

        self.N = 1

        self.R = R
        self.beta = beta
        self.name = "%d-Subjettiness" % self.N

        # Save a list of coordinates to approximate the shape
        self.t = self.sample(self.N)

    # Uniform sampling of anchor points 
    def sample(self, N):
        return tf.keras.backend.random_uniform(shape=(N,1), dtype=tf.float64)

    # Return actual points coordinates
    def get_points(self):
        return self.points

    def get_zs(self):
        return self.z 


    # Format parameters in a nice dictionary
    def get_param_dict(self, loss):

        dict = super().get_param_dict(loss)
        for i in range(self.N):
            dict["Center"] = self.points[i,:].numpy()
            dict["z"] = self.z.numpy()
        return dict

    # Initializer
    def initialize(self, event):

        self.z = tf.Variable(1.0 / self.N,
                             trainable = True,
                             dtype = tf.float64)


        if self.initializer == "kt":

            self.points = tf.Variable(kt_initializer(event, self.N, self.R),
                                trainable = True,
                                dtype = tf.float64)
            self.parameters = [self.points]

        else:
            self.points = tf.Variable(self.initializer,
                                trainable = True,
                                dtype = tf.float64)
            self.parameters = [self.points]

    def draw(self, ax):
        ax.scatter(self.points[:,0], self.points[:,1], color = "purple")

    def __str__(self):
        return ""



class NStructure(CustomObservable):

    def __init__(self, structure_class, N, name, initializer = None, z = 1.0, R = 0.5, beta = 1.0, **kwargs):

        self.shapes = []
        self.N = N
        for i in range(self.N):
            self.shapes.append(structure_class(initializer, z / self.N, R, beta, **kwargs))
        self.name = name
        self.initializer = initializer

        super().__init__(self.shapes, self.name, z = z, R = R, beta = beta, )


    def initialize(self, event):

        if self.initializer == "kt":
            points = kt_initializer(event, self.N, self.R)
        else:
            points = self.initializer

        for i in range(self.N):
            self.shapes[i].points = tf.Variable((points[i],),
                                trainable = True,
                                dtype = tf.float64)
            self.shapes[i].parameters = [self.shapes[i].points]



class NSubJettiness(Observable):

    def __init__(self, initializer = None, z = 1.0, N = 1, R = 0.5, beta = 1.0, ):
        super().__init__()

        # N jet parameters
        self.initializer = initializer
        if initializer != "kt":
            self.points = tf.Variable(initializer,
                                trainable = True,
                                dtype = tf.float64)
            self.parameters = [self.points]


        self.z = tf.Variable(tf.ones(shape = (N,), dtype = tf.float64) * z / N,
                             trainable = True,
                             dtype = tf.float64)

        self.N = N

        self.R = R
        self.beta = beta
        self.name = "%d-Subjettiness" % self.N

        # Save a list of coordinates to approximate the shape
        self.t = self.sample(N)

    # Uniform sampling of anchor points 
    def sample(self, N):
        return tf.keras.backend.random_uniform(shape=(N,1), dtype=tf.float64)

    # Return actual points coordinates
    def get_points(self):
        return self.points

    def get_zs(self):
        return self.z 


    # Format parameters in a nice dictionary
    def get_param_dict(self, loss):

        dict = super().get_param_dict(loss)
        for i in range(self.N):
            dict["Center_%d" % (i + 1)] = self.points[i,:].numpy()
            dict["z_%d" % (i + 1)] = self.z[i].numpy()
        return dict

    # Initializer
    def initialize(self, event):

        self.z = tf.Variable(tf.ones(shape = (self.N, ), dtype = tf.float64) * 1.0 / self.N,
                             trainable = True,
                             dtype = tf.float64)


        if self.initializer == "kt":

            self.points = tf.Variable(kt_initializer(event, self.N, self.R),
                                trainable = True,
                                dtype = tf.float64)
            self.parameters = [self.points]

        else:
            self.points = tf.Variable(self.initializer,
                                trainable = True,
                                dtype = tf.float64)
            self.parameters = [self.points]

    def draw(self, ax):
        ax.scatter(self.points[:,0], self.points[:,1], color = "purple")

    def __str__(self):
        return ""

class Thrust(Observable):

    def __init__(self, initializer, z = 1.0, R = 0.5, beta = 1.0):
        super().__init__()


        # N jet parameters
        self.point = tf.Variable(initializer,
                             trainable = True,
                             dtype = tf.float64)


        self.z = tf.Variable(z,
                             trainable = True,
                             dtype = tf.float64)

        self.parameters = [self.point]

        self.R = R
        self.beta = beta

        self.name = "Thrust"

        # Save a list of coordinates to approximate the shape
        self.t = self.sample()

    def initialize(self, event):
        pass
    
    def sample(self,):
        return tf.keras.backend.random_uniform(shape=(1,1), dtype=tf.float64)

    def get_points(self):
        return tf.concat( (self.point, -self.point))    

    def draw(self, ax):
        ax.scatter(self.point[:,0], self.point[:,1], color = "purple")
        ax.scatter(-1 * self.points[:,0], -1 * self.point[:,1], color = "purple")

    def __str__(self):
        return ""





class Circle(Observable):

    def __init__(self, center, radius, z = 1.0, R = 0.5, beta = 1.0, N = 100) -> None:
        super().__init__()

        # Circle parameters
        self.center_initializer = center
        self.radius_initializer = radius
        self.z_initializer = z

        self.initialize(None)
        self.parameters = [self.center, self.radius]
        self.N = N

        self.R = R
        self.beta = beta
        self.name = "Circle"

        # Save a list of coordinates to approximate the shape
        self.t = self.sample(N)


    def initialize(self, event):
        self.center = tf.Variable(self.center_initializer,
                             trainable = True,
                             dtype = tf.float64)
        self.radius = tf.Variable(self.radius_initializer,
                             trainable = True,
                             dtype = tf.float64)
        self.z = tf.Variable(self.z_initializer,
                             trainable = True,
                             dtype = tf.float64)
        self.parameters = [self.center, self.radius]


    def sample(self, N):
        return 2 * np.pi * tf.keras.backend.random_uniform(shape=(N,), dtype=tf.float64)

    def get_points(self):
        xy = tf.stack([tf.math.cos(self.t), tf.math.sin(self.t)], axis = 1)
        return self.center + self.radius * xy

    def get_param_dict(self, loss):
        dict =  super().get_param_dict(loss)
        dict["Center"] = self.center.numpy()
        dict["radius"] = tf.math.abs(self.radius.numpy())
        dict["z"] = self.z.numpy()
        return dict


    def draw(self, ax):
        draw_circle = pltCircle(self.center, self.radius, facecolor = None, edgecolor = "purple",  fill = False,)
        ax.add_artist(draw_circle)

    def __str__(self):
        return "Radius: %.3f, z: %.3f GeV" % (self.radius.numpy(), self.z.numpy())



class Disk(Observable):

    def __init__(self, center, radius, z = 1.0, R = 0.5, beta = 1.0, N = 100) -> None:
        super().__init__()

       # Circle parameters
        self.center_initializer = center
        self.radius_initializer = radius
        self.z_initializer = z

        self.initialize(None)
        self.parameters = [self.center, self.radius]
        self.N = N

        # Save a list of coordinates to approximate the shape
        self.name = "Disk"
        self.t = self.sample(N)
        
    def initialize(self, event):
        self.center = tf.Variable(self.center_initializer,
                             trainable = True,
                             dtype = tf.float64)
        self.radius = tf.Variable(self.radius_initializer,
                             trainable = True,
                             dtype = tf.float64)
        self.z = tf.Variable(self.z_initializer,
                             trainable = True,
                             dtype = tf.float64)
        self.parameters = [self.center, self.radius]

    def sample(self, N):
        phi = 2 * np.pi * tf.keras.backend.random_uniform(shape=(N,), dtype=tf.float64)
        r =  np.sqrt(tf.keras.backend.random_uniform(shape=(N,), dtype=tf.float64))
        return phi, r

    def get_param_dict(self, loss):
        dict =  super().get_param_dict(loss)
        dict["Center"] = self.center.numpy()
        dict["radius"] = tf.math.abs(self.radius.numpy())
        dict["z"] = self.z.numpy()
        return dict

    def get_points(self):
        self.phi, self.r = self.t
        xy = self.r[:,None] * tf.stack([tf.math.cos(self.phi), tf.math.sin(self.phi)], axis = 1)
        return self.center + tf.math.abs(self.radius) * xy


    def draw(self, ax):
        draw_circle = [pltCircle(self.center, tf.math.abs(self.radius), facecolor = None, edgecolor = "purple", zorder = 20)]
        p = PatchCollection(draw_circle, color = "purple", alpha = 0.25, zorder = 20)
        ax.add_collection(p)
    
    def __str__(self):
        return "Radius: %.3f, z: %.3f GeV" % (tf.math.abs(self.radius).numpy(), self.z.numpy())


class Gaussian(Observable):

    def __init__(self, center, radius, z = 1.0, R = 0.5, beta = 1.0, N = 100) -> None:
        super().__init__()

       # Circle parameters
        self.center_initializer = center
        self.radius_initializer = radius
        self.z_initializer = z

        self.initialize(None)
        self.parameters = [self.center, self.radius]
        self.N = N

        # Save a list of coordinates to approximate the shape
        self.name = "Gaussian"
        self.t = self.sample(N)
        
    def initialize(self, event):
        self.center = tf.Variable(self.center_initializer,
                             trainable = True,
                             dtype = tf.float64)
        self.radius = tf.Variable(self.radius_initializer,
                             trainable = True,
                             dtype = tf.float64)
        self.z = tf.Variable(self.z_initializer,
                             trainable = True,
                             dtype = tf.float64)
        self.parameters = [self.center, self.radius]

    def sample(self, N):
        t = tf.keras.backend.random_normal(shape = (N, 2), dtype = tf.float64)
        return t

    def get_param_dict(self, loss):
        dict =  super().get_param_dict(loss)
        dict["Mean"] = self.center.numpy()
        dict["Std"] = tf.math.abs(self.radius.numpy())
        dict["z"] = self.z.numpy()
        return dict

    def get_points(self):

        return self.center + self.t * self.radius


    def draw(self, ax):
        draw_circle = [pltCircle(self.center, tf.math.abs(self.radius), facecolor = None, edgecolor = "purple", zorder = 20)]
        p = PatchCollection(draw_circle, color = "purple", alpha = 0.25, zorder = 20)
        ax.add_collection(p)
    
    def __str__(self):
        return "Std: %.3f, z: %.3f GeV" % (tf.math.abs(self.radius).numpy(), self.z.numpy())




class Ellipse(Observable):

    def __init__(self, center, semi_major_radius, eccentricity = 0, angle = 0, z = 1.0, N = 100) -> None:
        super().__init__()

       # Circle parameters
        self.center_initializer = center
        self.radius_initializer = semi_major_radius
        self.eccentricity_initializer = eccentricity
        self.angle_initializer = angle
        self.z_initializer = z

        self.initialize(None)
        self.parameters = [self.center, self.radius, self.eccentricity, self.angle]
        self.N = N

        # Save a list of coordinates to approximate the shape
        self.name = "Disk"
        self.t = self.sample(N)
        
    def initialize(self, event):
        self.center = tf.Variable(self.center_initializer,
                             trainable = True,
                             dtype = tf.float64)
        self.radius = tf.Variable(self.radius_initializer,
                             trainable = True,
                             dtype = tf.float64)
        self.eccentricity = tf.Variable(self.radius_initializer,
                             trainable = True,
                             dtype = tf.float64)
        self.angle = tf.Variable(self.angle_initializer,
                             trainable = True,
                             dtype = tf.float64)
        self.z = tf.Variable(self.z_initializer,
                             trainable = True,
                             dtype = tf.float64)
        self.parameters = [self.center, self.radius, self.eccentricity, self.angle]

    def sample(self, N):
        phi = 2 * np.pi * tf.keras.backend.random_uniform(shape=(N,), dtype=tf.float64)
        r =  np.sqrt(tf.keras.backend.random_uniform(shape=(N,), dtype=tf.float64))
        return phi, r

    def get_param_dict(self, loss):
        dict =  super().get_param_dict(loss)
        dict["Center"] = self.center.numpy()
        dict["major_axis"] = np.abs(self.radius.numpy())
        dict["eccentricity"] = np.abs(self.eccentricity.numpy())
        if dict["eccentricity"] > 1.0:
            dict["eccentricity"] = 1.0 /   dict["eccentricity"]
        dict["angle"] = self.angle.numpy() % (2 * np.pi)
        dict["z"] = self.z.numpy()
        return dict

    def get_points(self):
        self.phi, self.r = self.t
        c = self.eccentricity * self.radius
        b = tf.math.sqrt(tf.math.abs(self.radius**2 - c**2))
        xy = self.r[:,None] * tf.stack([tf.math.abs(self.radius) * tf.math.cos(self.phi + self.angle), b * tf.math.sin(self.phi + self.angle)], axis = 1)
        return self.center + xy


    def draw(self, ax):
        c = self.eccentricity * self.radius
        b = tf.math.sqrt(tf.math.abs(self.radius**2 - c**2))
        e = [pltEllipse(self.center, tf.math.abs(self.radius).numpy(), b.numpy(), angle = self.angle.numpy() * 180 / 3.14159, color = "purple", edgecolor = "purple", alpha = 0.25, zorder = 20),]
        p = PatchCollection(e, color = "purple", alpha = 0.25, zorder = 20)
        ax.add_collection(p)
    
    def __str__(self):
        return r"Major Axis: %.3f, e: %.3f, $\theta$: %.3f, z: %.3f GeV" % (tf.math.abs(self.radius).numpy(), self.get_param_dict(1.0)["eccentricity"], self.angle % (2 * np.pi),  self.z.numpy())






class Uniform(Observable):

    def __init__(self, lower_left, upper_right, trainable_z = True, z = 1.0, R = 1.0, beta = 1.0,  N = 100):
        super().__init__()

        # Background parameters
        self.lower_left = tf.Variable(lower_left,
                                trainable = False,
                                dtype = tf.float64)
        
        self.upper_right = tf.Variable(upper_right,
                                trainable = False,
                                dtype = tf.float64)

        # Circle parameters
        self.z_initializer = z

        self.initialize(None)
        self.z = tf.Variable(z,
                             trainable = trainable_z,
                             dtype = tf.float64)
        self.parameters = []
        self.N = N

        self.R = R
        self.beta = beta

        # Save a list of coordinates to approximate the shape
        self.name = "Uniform"
        self.t = self.sample(N)

    def sample(self, N):
        return tf.keras.backend.random_uniform(shape=(N,2), dtype=tf.float64)

    def get_points(self):
        return self.lower_left + (self.upper_right - self.lower_left) * self.t

    def get_param_dict(self, loss):
        dict =  super().get_param_dict(loss)
        dict["z"] = self.z.numpy()
        return dict


    def draw(self, ax):
        pass

    def initialize(self, event):
        self.z = tf.Variable(self.z_initializer,
                             trainable = True,
                             dtype = tf.float64)
        
    def __str__(self):
        return "Background: %.3f GeV" % (self.z.numpy())



# ##################################
# ########## INITIALIZERS ##########
# ##################################

def kt_initializer(event, N, R):

    y, z = event

    four_vectors = []
    for (y_i, z_i) in zip(y, z):
        v = (z_i, y_i[0], y_i[1], 0)
        four_vectors.append(v)
    four_vectors = np.array(four_vectors, dtype = [("pt", "f8"),("eta", "f8"),("phi", "f8"),("mass", "f8")])
    sequence = cluster(four_vectors, R=R, p=1)
    jets = sequence.exclusive_jets(N)

    # Apply initialization
    jets = jets[:N]
    initialization = []
    for jet in jets:
        initialization.append([jet.eta, jet.phi])
    initialization = np.array(initialization).astype(np.float32)
    return initialization