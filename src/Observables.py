# Standard stuff
from abc import ABC, abstractmethod
from matplotlib.pyplot import axis
from matplotlib.patches import Circle as pltCircle
from matplotlib.patches import Polygon as pltPolygon
from matplotlib.collections import PatchCollection
import numpy as np
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.shaper import Shaper
from src.utils.plot_utils import plot, make_gif

import tensorflow as tf









class Observable(ABC):

    def __init__(self) -> None:
        super().__init__()
        self.shapes = [self,]
        self.R = None
        self.beta = None


    @abstractmethod
    # A function to generate a list of points to approximate the shape
    def sample(self, N):
        pass

    @abstractmethod
    # Generate points according to the parameters
    def get_points(self,):
        pass

    @abstractmethod    
    def get_param_dict(self, loss):
        return {"EMD" : loss}

    def get_zs(self, ):
        return tf.ones(shape = (self.N,), dtype = tf.float64) * self.z / self.N

    @abstractmethod
    def draw(self, ax):
        pass

    @abstractmethod
    def __str__(self):
        pass


    @abstractmethod
    def initialize(self):
        pass


    # Calculate Observable on an event (y, z)
    def calculate(self, event, training_config_dict, gif_filename = None, plot_filename = None):

        y_i, z_i = event
        filenames = []

        # Set up training
        self.initialize(event)
        shaper = Shaper(self.shapes, R = self.R, beta = self.beta) # TODO: Remove shape array
        epochs = training_config_dict["epochs"]
        optimizer = training_config_dict["optimizer"]
        verbose = training_config_dict["verbose"]
        resample = training_config_dict["resample"]
        z_schedule = 0.5
        if "z_schedule" in training_config_dict:
            z_schedule = training_config_dict["z_schedule"]
        shaper.compile(optimizer = optimizer)

        l_0 = np.inf
        x_ij_0 = None
        params = {}
        stopping_counter = 0


        half_epochs = int(epochs * z_schedule)
        train_z = False

        # Train
        epoch = 0
        for epoch in range(half_epochs):


            if resample:
                for shape in self.shapes:
                    shape.t = shape.sample(shape.N)
            l, x_ij, y_j, z_j, _ = shaper.train_step((y_i, z_i), False)
            if verbose:
                print("Epoch %d: %.3f, z0 = %.3f" % (epoch, l, self.z))

            if gif_filename is not None:

                rad = self.R
                fname = gif_filename + f"_{epoch}.png"
                filenames.append(fname)
                y_j, z_j = shaper.get_samples()
                plot(y_i, z_i, y_j, z_j, x_ij, l, rad, self.z,  fname, self.name + " Epoch %d" % epoch, self.shapes)

            # Get best result so far
            if l < l_0:
                l_0 = l
                x_ij_0 = x_ij
                params = self.get_param_dict(l_0.numpy())
                stopping_counter = 0
            else:
                stopping_counter += 1

            # Halfway through, turn on z training
            if epoch == half_epochs:
                train_z = True
                stopping_counter = 0
            
            if "early_stop" in training_config_dict and stopping_counter >= training_config_dict["early_stop"] and not train_z:
                epoch = half_epochs - 1

            # Early stopping
            if "early_stop" in training_config_dict and stopping_counter >= training_config_dict["early_stop"]: 
                break


        stopping_counter = 0
        for epoch in range(half_epochs, epochs):


            if resample:
                for shape in self.shapes:
                    shape.t = shape.sample(shape.N)
            l, x_ij, y_j, z_j, _ = shaper.train_step((y_i, z_i), True)
            if verbose:
                print("Epoch %d: %.3f, z0 = %.3f" % (epoch, l, self.z))

            if gif_filename is not None:

                rad = self.R
                fname = gif_filename + f"_{epoch}.png"
                filenames.append(fname)
                y_j, z_j = shaper.get_samples()
                plot(y_i, z_i, y_j, z_j, x_ij, l, rad, self.z,  fname, self.name + " Epoch %d" % epoch, self.shapes)

            # Get best result so far
            if l < l_0:
                l_0 = l
                x_ij_0 = x_ij
                params = self.get_param_dict(l_0.numpy())
                stopping_counter = 0
            else:
                stopping_counter += 1

            # # Halfway through, turn on z training
            # if epoch == half_epochs:
            #     train_z = True
            #     stopping_counter = 0
            
            # if "early_stop" in training_config_dict and stopping_counter >= training_config_dict["early_stop"] and not train_z:
            #     epoch = half_epochs - 1

            # Early stopping
            if "early_stop" in training_config_dict and stopping_counter >= training_config_dict["early_stop"]: 
                break

            epoch += 1


        if gif_filename is not None:
            make_gif(filenames, gif_filename)

        if plot_filename is not None:
            plot(y_i, z_i, y_j, z_j, x_ij, l, self.R, self.z,  plot_filename, self.name, self.shapes)


        return params



    
    # # Calculate Observable on an event (y, z)
    # def __call__(self, *args: Any, **kwds: Any) -> Any:
    #     return super().__call__(*args, **kwds))

        

class CustomObservable(Observable):

    def __init__(self, shapes, name, ids = None, z = 1.0, R = 0.8, beta = 1.0):
        
        self.shapes = shapes
        self.R = R
        self.beta = beta
        self.z = shapes[0].z
        self.name = name

        if ids is None:
            self.ids = np.arange(len(shapes))
        else: 
            self.ids = ids

    
    def initialize(self, event):
        for shape in self.shapes:
            shape.initialize(event)
        self.z = self.shapes[0].z

    def sample(self):
        pass


    def get_param_dict(self, loss):
        dict = super().get_param_dict(loss)
        for (i,shape) in enumerate(self.shapes):
            dict_i = shape.get_param_dict(loss)
            for key, value in dict_i.items():
                if key != "EMD":
                    dict[key + "_%d" % self.ids[i]] = value 
        return dict



    # Generate points according to the parameters
    def get_points(self,):
        pass

    def draw(self, ax):
        pass

    def __str__(self):
        pass