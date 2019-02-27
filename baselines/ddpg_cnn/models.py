import tensorflow as tf
from baselines.common.models import get_network_builder

def cnn(image):
    h = tf.layers.conv2d(image, 16, 5, strides=(1,2), activation=tf.nn.relu, name="conv1") #57x157x16
    h = tf.layers.conv2d(h, 16, 5, strides=(1,2), activation=tf.nn.relu, name="conv2")    #53x77x16
    h = tf.layers.conv2d(h, 24, 5, strides=2, activation=tf.nn.relu, name="conv3")        #25x37x24
    h = tf.layers.conv2d(h, 32, 5, strides=2, activation=tf.nn.relu, name="conv4")        #11x17x32
    h = tf.layers.conv2d(h, 64, 5, strides=2, activation=tf.nn.relu, name="conv5")        #4x7x64
    return tf.reshape(h, [-1, 4*7*64]) #Flatten Tensor -> #1792

class Model(object):
    def __init__(self, name, latent_size, network='mlp', **network_kwargs):
        self.name = name
        self.latent_size = latent_size
        self.network_builder = get_network_builder(network)(**network_kwargs)

    @property
    def vars(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)

    @property
    def trainable_vars(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

    @property
    def perturbable_vars(self):
        return [var for var in self.trainable_vars if 'LayerNorm' not in var.name]


class Actor(Model):
    def __init__(self, nb_actions, latent_size, name='actor', network='mlp', **network_kwargs):
        super().__init__(name=name, network=network, latent_size=latent_size, **network_kwargs)
        self.nb_actions = nb_actions

    def __call__(self, obs, additional, reuse=False):
        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            image = cnn(obs)
            x = tf.layers.dense(image, self.latent_size, name="fc1")
            x = tf.concat([x, additional], 1)
            x = self.network_builder(x)
            x = tf.layers.dense(x, self.nb_actions, kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3), name="fc2")
            x = tf.nn.tanh(x)
        return x


class Critic(Model):
    def __init__(self, latent_size, name='critic', network='mlp', **network_kwargs):
        super().__init__(name=name, network=network, latent_size=latent_size, **network_kwargs)
        self.layer_norm = True

    def __call__(self, obs, additional, action, reuse=False):
        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            image = cnn(obs)
            x = tf.layers.dense(image, self.latent_size, name="fc1")
            x = tf.concat([x, additional, action], 1)
            x = self.network_builder(x)
            x = tf.layers.dense(x, 1, kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3), name='output')
        return x

    @property
    def output_vars(self):
        output_vars = [var for var in self.trainable_vars if 'output' in var.name]
        return output_vars
