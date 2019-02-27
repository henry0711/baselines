import tensorflow as tf
from baselines.common.models import get_network_builder

class Encoder(object):
    def __init__(self, name, latent_size):
        self.name = name
        self.latent_size = latent_size

    @property
    def vars(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)

    @property
    def trainable_vars(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

    @property
    def perturbable_vars(self):
        return [var for var in self.trainable_vars if 'LayerNorm' not in var.name]

    def __call__(self, obs, epsilon, reuse=False):
        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            h = tf.layers.conv2d(obs, 16, 5, strides=(1,2), activation=tf.nn.relu, name="conv1")  #57x157x16
            h = tf.layers.conv2d(h, 16, 5, strides=(1,2), activation=tf.nn.relu, name="conv2")    #53x77x16
            h = tf.layers.conv2d(h, 24, 5, strides=2, activation=tf.nn.relu, name="conv3")        #25x37x24
            h = tf.layers.conv2d(h, 32, 5, strides=2, activation=tf.nn.relu, name="conv4")        #11x17x32
            h = tf.layers.conv2d(h, 64, 5, strides=2, activation=tf.nn.relu, name="conv5")        #4x7x64
            h = tf.reshape(h, [-1, 4*7*64]) #Flatten Tensor -> #1792
            mu = tf.layers.dense(h, self.latent_size, name="fc_mu")
            logvar = tf.layers.dense(h, self.latent_size, name="fc_log_var")
            sigma = tf.exp(logvar / 2.0)
            #epsilon = tf.random_normal([batch_size, self.latent_size])
            z = mu + sigma * epsilon
        return z, mu, logvar


class Decoder(object):
    def __init__(self, name):
        self.name = name

    @property
    def vars(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)

    @property
    def trainable_vars(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

    @property
    def perturbable_vars(self):
        return [var for var in self.trainable_vars if 'LayerNorm' not in var.name]

    def __call__(self, z, reuse=False):
        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            h = tf.layers.dense(z, 4*7*64, name="fc")
            h = tf.reshape(h, [-1, 4, 7, 64])
            h = tf.layers.conv2d_transpose(h, 32, 5, strides=2, activation=tf.nn.relu, name="deconv1")
            h = tf.layers.conv2d_transpose(h, 24, 5, strides=2, activation=tf.nn.relu, name="deconv2")
            h = tf.layers.conv2d_transpose(h, 16, 5, strides=2, activation=tf.nn.relu, name="deconv3")
            h = tf.layers.conv2d_transpose(h, 16, 5, strides=(1,2), activation=tf.nn.relu, name="deconv4")
            y = tf.layers.conv2d_transpose(h, 3, 5, strides=(1,2), activation=tf.nn.sigmoid, name="deconv5")
        return y


class Model(object):
    def __init__(self, name, network='mlp', **network_kwargs):
        self.name = name
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
    def __init__(self, nb_actions, name='actor', network='mlp', **network_kwargs):
        super().__init__(name=name, network=network, **network_kwargs)
        self.nb_actions = nb_actions

    def __call__(self, obs, reuse=False):
        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            x = self.network_builder(obs)
            x = tf.layers.dense(x, self.nb_actions, kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3))
            x = tf.nn.tanh(x)
        return x


class Critic(Model):
    def __init__(self, name='critic', network='mlp', **network_kwargs):
        super().__init__(name=name, network=network, **network_kwargs)
        self.layer_norm = True

    def __call__(self, obs, action, reuse=False):
        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            x = tf.concat([obs, action], axis=-1) # this assumes observation and action can be concatenated
            x = self.network_builder(x)
            x = tf.layers.dense(x, 1, kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3), name='output')
        return x

    @property
    def output_vars(self):
        output_vars = [var for var in self.trainable_vars if 'output' in var.name]
        return output_vars
