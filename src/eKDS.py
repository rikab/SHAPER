# #############################
# ########## IMPORTS ##########
# #############################

# Standard stuff
from matplotlib.pyplot import axis
from matplotlib.patches import Polygon, Circle
from matplotlib.collections import PatchCollection
import numpy as np
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ML stuff
import keras
from keras.models import Model
from keras.layers import Dense, Dropout, Input, Concatenate
import tensorflow as tf

import energyflow as ef
from src.plot_utils import plot, make_gif

# ############################
# ########## LOSSES ##########
# ############################

@tf.function
def emd_loss(y_i, y_j, x_ij, R, beta = 1.0):



    theta_ij = tf.reduce_sum(tf.abs(tf.math.pow((y_i[:,None,:] - y_j[None,:,:] ) / R, beta)), axis = 2)
    return  tf.math.pow(tf.reduce_sum(theta_ij * x_ij), 1.0 / 1.0) 


# @tf.function
def emd(y_i, z_i, y_j, z_j, R):

    # Format events
    ev0 = tf.concat((tf.expand_dims(z_i,1), y_i), axis  = 1)
    ev1 = tf.concat((tf.expand_dims(z_j,1), y_j), axis  = 1)
    # ev0[:,0] = z_i
    # ev1[:,0] = z_j
    # ev0[:,1:] = y_i
    # ev1[:,1:] = y_j

    emd, G = ef.emd.emd(ev0, ev1, R=R, return_flow = True, beta = 1.0, norm = True)
    # print(tf.reduce_sum(G[-1,:]))
    return G[:z_i.shape[0], :z_j.shape[0]], emd

# ################################
# ########## eKDS MODEL ##########
# ################################

class polygonal_sample(tf.keras.layers.Layer):
    """ Samples N points uniformly from a polygon (poly, N)"""

    def call(self, inputs):
        polygon, N = inputs
        num_vertices, dim = polygon.shape[0], polygon.shape[1]
        
        t_vector = tf.linalg.normalize(
            tf.keras.backend.random_uniform(shape=(N,num_vertices), dtype=tf.float64), ord = 1, axis=1
        )[0]

        return tf.reduce_sum(t_vector[:,:,None] * polygon[None,:,:], axis = 1)


class polygonal_sampling_vector(tf.keras.layers.Layer):
    """ Samples N points uniformly from a polygon (poly, N)"""

    def call(self, inputs):

        polygon, N, dim = inputs
        num_vertices = polygon.shape[0]
        
        t_vector = tf.linalg.normalize(
            tf.keras.backend.random_uniform(shape=(N,num_vertices), dtype=tf.float64), ord = 1, axis=1
        )[0]

        return t_vector


# Model
class eKDS(tf.keras.Model):


    def __init__(self, shapes, R, **kwargs):
        
        super(eKDS, self).__init__(**kwargs)
     
        self.shapes = shapes
        self.R = R
        self.z = []
        for shape in shapes:
            self.z.append(shape.z)

    
    # def __init__(self, graph, A_initializer, z_initializer = 0.0, dim = 2, R = 1, samples = 10, resample = False, **kwargs):

    #     # Parameters
    #     super(eKDS, self).__init__(**kwargs)


    #     # TODO: Add verifier to check that initializer shape matches structure

    #     self.A_initializer = A_initializer
    #     self.A = tf.Variable(A_initializer,
    #                          trainable = True,
    #                          dtype = tf.float64)
    #     self.z_0 = tf.Variable(z_initializer, trainable = False, dtype = tf.float64)
    #     self.R = R
    #     self.graph = graph
    #     self.structure = graph.structure





    # @tf.function
    def get_samples(self, ):

        points = []
        zs = []
        for shape in self.shapes:
            points.append(shape.get_points())
            zs.append(shape.get_zs())
        return tf.concat(points, axis = 0), tf.concat(zs, axis = 0)

        if self.resample:
            return self.polygonal_sample((self.A, N)), tf.ones(shape=(N), dtype = tf.float64) * (1-self.z_0)/ N
        else:
            points = []
            for i, t in enumerate(self.t_vectors):
                indices = self.structure[i]
                polygon = tf.gather(self.A, indices)
                points.append(tf.reduce_sum(t[:,:,None] * polygon[None,:,:], axis = 1) )
            return tf.concat(points, axis = 0), tf.ones(shape=(N * self.num_polygons), dtype = tf.float64) * (1-self.z_0) / (N * self.num_polygons)

    # @tf.function
    def encode(self, y_i, z_i, a_j = None, z_j = None):

        if (a_j is None or z_j is None):
            a_j, z_j = self.get_samples()
        return emd(y_i, z_i, a_j, z_j, self.R)


    def reset(self):
        """
        Reset polygon vertices to initializer values, and resample points before a new training
        """
        pass

    # @tf.function()
    def train_step(self, inputs, train_z = False, epsilon = 0.001):

        # Process inputs0
        y_i, z_i = inputs

        # Calculate the loss and gradients for shape parameters
        with tf.GradientTape(persistent = True) as tape:
            y_j, z_j = self.get_samples()
            # print (tf.reduce_sum(z_j))
            x_ij, l = self.encode(y_i, z_i, y_j, z_j)
            loss = emd_loss(y_i, y_j, x_ij, self.R)

        # Compute gradients and update shape parameters
        for shape in self.shapes:
            gradients = tape.gradient(loss, [*shape.parameters,])
            self.optimizer.apply_gradients(zip(gradients, [*shape.parameters,]))

        self.compiled_metrics.update_state(y_i, z_i)


        # Compute z gradients by hand
        if train_z and len(self.shapes) > 1:
            gradients = []

            for shape in self.shapes:
                shape.z = shape.z + epsilon
                y_j, z_j = self.get_samples()
                z_j = z_j / tf.reduce_sum(z_j)
                x_ij, l = self.encode(y_i, z_i, y_j, z_j)
                loss2 = emd_loss(y_i, y_j, x_ij, self.R)
                gradients.append((loss2-loss)/epsilon)
                shape.z = shape.z - epsilon
            # gradients.append(-1 * tf.reduce_sum(gradients))


            # Update z's, then normalize
            self.optimizer.apply_gradients(zip(gradients, self.z))
            zs = []
            for i, shape in enumerate(self.shapes):
                zs.append(self.z[i].numpy())
            zs = np.array(zs).reshape((1, -1))
            zs = activate(zs)
            for i,shape in enumerate(self.shapes):
                shape.z = zs[0,i]

        return loss, x_ij, y_j, z_j, {m.name: m.result() for m in self.metrics}

    def train(self, y_i, z_i, epochs, lr = 0.0025, verbose = True, train_schedule = 0.5, flavor_text = None, gif_dict = None):

        # Reset architecture for clean run
        self.reset()
        self.compile(optimizer = tf.keras.optimizers.Adam(lr = lr))
        l_prev = np.inf
        early_stopping_counter = 0
        
        # Training gif setup
        filenames = []
        
        for i in range(0, int(epochs * train_schedule)):

            l, x_ij, y_j, z_j, _ = self.train_step((y_i, z_i), False)
            
            if l_prev <= l:
                early_stopping_counter += 1
            else:
                early_stopping_counter = 0
                l_prev = np.copy(l)
            if early_stopping_counter >= 50:
                break
            if verbose:
                if flavor_text is not None:
                    print("%s: Epoch %d: %.3f, z0 = %.3f" % (flavor_text, i, l, self.shapes[0].z))
                else:
                    print("Epoch %d: %.3f, z0 = %.3f" % (i, l, self.shapes[0].z))
 
            if gif_dict is not None:


                rad = gif_dict["R"]
                fname = gif_dict["filename"] + f"_{i}.png"
                filenames.append(fname)
                y_j, z_j = self.get_samples()
                plot(y_i, z_i, y_j, z_j, x_ij, l, rad, self.shapes[0].z,  fname, self.shapes)


        early_stopping_counter = 0
        for i in range(int(epochs * train_schedule), epochs):

            l, x_ij, y_j, z_j, _ = self.train_step((y_i, z_i), True)

            if l_prev <= l:
                early_stopping_counter += 1
            else:
                early_stopping_counter = 0
                l_prev = np.copy(l)
            if early_stopping_counter >= 50:
                break
            if verbose:
                if flavor_text is not None:
                    print("%s: Epoch %d: %.3f, z0 = %.3f" % (flavor_text, i, l, self.shapes[0].z))
                else:
                    print("Epoch %d: %.3f, z0 = %.3f" % (i, l, self.shapes[0].z))


         
            if gif_dict is not None:

                rad = gif_dict["R"]
                fname = gif_dict["filename"] + f"_{i}.png"
                filenames.append(fname)
                y_j, z_j = self.get_samples()
                plot(y_i, z_i, y_j, z_j, x_ij, l, rad, self.shapes[0].z,  fname, self.shapes)
                
        
        if gif_dict is not None:
            make_gif(filenames, gif_dict["filename"])


        y_j, z_j = self.get_samples()
        x_ij, l = self.encode(y_i, z_i)
        print("Epoch %d: %.3f, z0 = %.3f" % (epochs, l, self.shapes[0].z))
        return l, x_ij


    def determine_z(self, y_i, z_i, y_j):

        theta_ij = tf.reduce_sum(tf.abs(tf.math.pow(y_i[:,None,:] - y_j[None,:,:], 1.0)), axis = 2) / self.R
        less_than_R = tf.cast(theta_ij <= 1.0, tf.int32) 
        indices = tf.cast(tf.reduce_sum(less_than_R, axis = 1) > 0, tf.float64)
        self.z_0 = 1 - tf.reduce_sum(indices * z_i)


    def get_polygons(self):
        return self.A



# @tf.function
def activate(f):
    
    m, n = f.shape
    cnt_m = tf.range(m)
    cnt_n = tf.range(n, dtype = tf.float64)
    u = tf.reverse(tf.sort(f, axis=1), axis = (1,))

    v = (tf.cumsum(u, axis=1) - 1) / (cnt_n + 1)
    indices = tf.stack((cnt_m, tf.reduce_sum( tf.cast(u > v, tf.int32), axis=1)-1), axis = -1)
    w = tf.gather(v, tf.reduce_sum( tf.cast(u > v, tf.int32), axis=1)-1, axis = 1, batch_dims = 1)
    # w = v[:, indices, axis=1) - 1]
    return tf.nn.relu(f - tf.reshape(w,(m,1)))


# ##############################
# ########## POLYGONS ##########
# ##############################

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
                patches.append(Polygon(A[polygon], True))
        p = PatchCollection(patches, color = "purple", alpha = 0.25)
        ax.add_collection(p)


