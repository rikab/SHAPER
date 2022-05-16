# #############################
# ########## IMPORTS ##########
# #############################

import numpy as np
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tensorflow as tf
import energyflow as ef

from src.utils.generic_utils import iter_or_rep


# #########################
# ########## EMD ##########
# #########################

# Explicit function for finding theta_ij gradients given a transfer matrix
@tf.function
def emd_loss(y_i, y_j, x_ij, R, beta = 1.0):

    theta_ij = tf.reduce_sum(tf.abs(tf.math.pow((y_i[:,None,:] - y_j[None,:,:] ) / R, beta)), axis = 2)
    return  tf.reduce_sum(theta_ij * x_ij) 


# Prebuilt EMD to find transfer matrix
def emd(y_i, z_i, y_j, z_j, R, beta = 1.0):

    # Format events
    temp = tf.reshape(z_j, [-1, 1])
    ev0 = tf.concat((tf.expand_dims(z_i,1), y_i), axis  = 1)
    ev1 = tf.concat((temp, y_j), axis  = 1)

    emd, G = ef.emd.emd(ev0, ev1, R=R, return_flow = True, beta = beta, norm = False)
    return G[:z_i.shape[0], :temp.shape[0]], emd



# ###########################
# ########## MODEL ##########
# ###########################

class Shaper(tf.keras.Model):

    def __init__(self, shapes, R, beta,  **kwargs) -> None:
        
        super(Shaper, self).__init__(**kwargs)

        self.shapes = shapes
        self.z = []
        self.R = R
        self.beta = beta


        for shape in shapes:
            self.z.append(shape.z)
            # if tf.shape(shape.z)[0] > 1:
            #     for i in range(tf.shape(shape.z)[0]):
            #         self.z.append(shape.z[i])
        self.z = np.array(self.z)

    # Sample shape(s) to get point distribution
    def get_samples(self, ):

        # TOOD: remove shape loop
        points = []
        zs = []
        for shape in self.shapes:
            points.append(shape.get_points())
            zs.append(shape.get_zs())

        return tf.concat(points, axis = 0), tf.concat(zs, axis = 0)


    # Use EMD to find trnsfer matrix
    def encode(self, y_i, z_i, a_j = None, z_j = None):

        if (a_j is None or z_j is None):
            a_j, z_j = self.get_samples()
        return emd(y_i, z_i, a_j, z_j, self.R, self.beta)



    def train_step(self, event, train_z =  False, epsilon = 0.001):

        # Format inputs
        y_i, z_i = event

        # Calculate the loss and gradients for shape parameters
        with tf.GradientTape(persistent = True) as tape:
            y_j, z_j = self.get_samples()
            # print (tf.reduce_sum(z_j))
            x_ij, l = self.encode(y_i, z_i, y_j, z_j)
            loss = emd_loss(y_i, y_j, x_ij, self.R, self.beta)

        # Compute gradients and update shape parameters
        for shape in self.shapes:
            gradients = tape.gradient(loss, [*shape.parameters,])
            self.optimizer.apply_gradients(zip(gradients, [*shape.parameters,]))

        self.compiled_metrics.update_state(y_i, z_i)


        # Compute z gradients by hand
        if train_z and len(self.shapes) > 1:
            gradients = []

            for shape in self.shapes:
    

                    # shape.z[i] = shape.z[i] + epsilon
                    shape.z = shape.z + epsilon
                    y_j, z_j = self.get_samples()
                    z_j = z_j / tf.reduce_sum(z_j)
                    x_ij, l = self.encode(y_i, z_i, y_j, z_j)
                    loss2 = emd_loss(y_i, y_j, x_ij, self.R, self.beta)
                    gradients.append((loss2-loss)/epsilon)
                    shape.z = shape.z - epsilon


            # Update z's, then normalize
            self.optimizer.apply_gradients(zip(gradients, self.z))
            zs = []
            for i, shape in enumerate(self.shapes):
                zs.append(self.z[i].numpy())
            zs = np.array(zs).reshape((1, -1))
            zs = activate(zs)
            # print(zs)
            for i,shape in enumerate(self.shapes):
                # for i in range(tf.shape(shape.z)[0]):
                shape.z = tf.Variable(zs[0,i])

        return loss, x_ij, y_j, z_j, {m.name: m.result() for m in self.metrics}






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