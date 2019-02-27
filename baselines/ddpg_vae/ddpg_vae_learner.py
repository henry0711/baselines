from copy import copy
from functools import reduce

import numpy as np
import tensorflow as tf
import tensorflow.contrib as tc

from baselines import logger
from baselines.common.tf_util import load_variables, save_variables
from baselines.common.mpi_adam import MpiAdam
from baselines.ddpg_vae.models import Encoder, Decoder
import baselines.common.tf_util as U
from baselines.common.mpi_running_mean_std import RunningMeanStd
try:
    from mpi4py import MPI
except ImportError:
    MPI = None

def normalize(x, stats):
    if stats is None:
        return x
    return (x - stats.mean) / stats.std


def denormalize(x, stats):
    if stats is None:
        return x
    return x * stats.std + stats.mean

def reduce_std(x, axis=None, keepdims=False):
    return tf.sqrt(reduce_var(x, axis=axis, keepdims=keepdims))

def reduce_var(x, axis=None, keepdims=False):
    m = tf.reduce_mean(x, axis=axis, keepdims=True)
    devs_squared = tf.square(x - m)
    return tf.reduce_mean(devs_squared, axis=axis, keepdims=keepdims)

def get_target_updates(vars, target_vars, tau):
    logger.info('setting up target updates ...')
    soft_updates = []
    init_updates = []
    assert len(vars) == len(target_vars)
    for var, target_var in zip(vars, target_vars):
        logger.info('  {} <- {}'.format(target_var.name, var.name))
        init_updates.append(tf.assign(target_var, var))
        soft_updates.append(tf.assign(target_var, (1. - tau) * target_var + tau * var))
    assert len(init_updates) == len(vars)
    assert len(soft_updates) == len(vars)
    return tf.group(*init_updates), tf.group(*soft_updates)


def get_perturbed_actor_updates(actor, perturbed_actor, param_noise_stddev):
    assert len(actor.vars) == len(perturbed_actor.vars)
    assert len(actor.perturbable_vars) == len(perturbed_actor.perturbable_vars)

    updates = []
    for var, perturbed_var in zip(actor.vars, perturbed_actor.vars):
        if var in actor.perturbable_vars:
            logger.info('  {} <- {} + noise'.format(perturbed_var.name, var.name))
            updates.append(tf.assign(perturbed_var, var + tf.random_normal(tf.shape(var), mean=0., stddev=param_noise_stddev)))
        else:
            logger.info('  {} <- {}'.format(perturbed_var.name, var.name))
            updates.append(tf.assign(perturbed_var, var))
    assert len(updates) == len(actor.vars)
    return tf.group(*updates)


class DDPG(object):
    def __init__(self, actor, critic, memory, observation_shape, action_shape, param_noise=None, action_noise=None,
        gamma=0.99, tau=0.001, beta=1.0, normalize_returns=False, enable_popart=False, normalize_observations=True,
        batch_size=128, observation_range=(-5., 5.), action_range=(-1., 1.), return_range=(-np.inf, np.inf),
        critic_l2_reg=0., actor_lr=1e-4, critic_lr=1e-3, clip_norm=None, reward_scale=1., epsilon=1e-3, latent_size=256,
        image_shape=(61, 317, 3)):
        # Inputs.
        self.obs0 = tf.placeholder(tf.float32, shape=(None,) + image_shape, name='obs0')
        self.obs1 = tf.placeholder(tf.float32, shape=(None,) + image_shape, name='obs1')
        self.additional0 = tf.placeholder(tf.float32, shape=(None, 2), name='additional_0')
        self.additional1 = tf.placeholder(tf.float32, shape=(None, 2), name='additional_1')
        self.terminals1 = tf.placeholder(tf.float32, shape=(None, 1), name='terminals1')
        self.rewards = tf.placeholder(tf.float32, shape=(None, 1), name='rewards')
        self.actions = tf.placeholder(tf.float32, shape=(None,) + action_shape, name='actions')
        self.param_noise_stddev = tf.placeholder(tf.float32, shape=(), name='param_noise_stddev')
        self.epsilon_ph = tf.placeholder(tf.float32, shape=(None, latent_size), name='epsilon')

        # Parameters.
        self.gamma = gamma
        self.tau = tau
        self.beta = beta
        self.memory = memory
        self.normalize_observations = normalize_observations
        self.normalize_returns = normalize_returns
        self.action_noise = action_noise
        self.param_noise = param_noise
        self.action_range = action_range
        self.return_range = return_range
        self.observation_range = observation_range
        self.critic = critic
        self.actor = actor
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.clip_norm = clip_norm
        self.enable_popart = enable_popart
        self.reward_scale = reward_scale
        self.batch_size = batch_size
        self.stats_sample = None
        self.critic_l2_reg = critic_l2_reg
        self.epsilon = epsilon
        self.image_shape = image_shape
        self.latent_size = latent_size
        self.kl_tolerance = 0.5
        
        # Observation normalization.
        if self.normalize_observations:
            with tf.variable_scope('obs_rms'):
                self.obs_rms = RunningMeanStd(shape=image_shape)
        else:
            self.obs_rms = None
            
        self.normalized_obs0 = tf.clip_by_value(normalize(self.obs0, self.obs_rms),
            self.observation_range[0], self.observation_range[1])
        self.normalized_obs1 = tf.clip_by_value(normalize(self.obs1, self.obs_rms),
            self.observation_range[0], self.observation_range[1])

        # Return normalization.
        if self.normalize_returns:
            with tf.variable_scope('ret_rms'):
                self.ret_rms = RunningMeanStd()
        else:
            self.ret_rms = None
            
        # Create encoder and decoder
        self.encoder = Encoder(name="encoder", latent_size=self.latent_size)
        self.decoder = Decoder(name="decoder")
        self.z, self.mu, self.logvar = self.encoder(self.normalized_obs0, epsilon=self.epsilon_ph)
        self.z1, self.mu1, self.logvar1 = self.encoder(self.normalized_obs1, epsilon=self.epsilon_ph, reuse=True)
        self.latent0 = tf.concat([self.z, self.additional0], 1)
        self.latent1 = tf.concat([self.z1, self.additional1], 1)
        self.y = self.decoder(self.z)

        # Create target networks.
        target_actor = copy(actor)
        target_actor.name = 'target_actor'
        self.target_actor = target_actor
        target_critic = copy(critic)
        target_critic.name = 'target_critic'
        self.target_critic = target_critic

        # Create networks and core TF parts that are shared across setup parts.
        self.actor_tf = actor(self.latent0)
        self.normalized_critic_tf = critic(self.latent0, self.actions)
        self.critic_tf = denormalize(tf.clip_by_value(self.normalized_critic_tf, self.return_range[0], self.return_range[1]), self.ret_rms)
        self.normalized_critic_with_actor_tf = critic(self.latent0, self.actor_tf, reuse=True)
        self.critic_with_actor_tf = denormalize(tf.clip_by_value(self.normalized_critic_with_actor_tf, self.return_range[0], self.return_range[1]), self.ret_rms)
        Q_obs1 = denormalize(target_critic(self.latent1, target_actor(self.latent1)), self.ret_rms)
        self.critic_target = self.rewards + (1. - self.terminals1) * gamma * Q_obs1

        # Set up parts.
        if self.param_noise is not None:
            self.setup_param_noise(self.latent0)
        self.setup_actor_optimizer()
        self.setup_critic_optimizer()
        self.setup_vae_optimizer()
        if self.normalize_returns and self.enable_popart:
            self.setup_popart()
        self.setup_stats()
        self.setup_target_network_updates()

        self.initial_state = None # recurrent architectures not supported yet
        
        #TF summaries (TF Board)
        tf.summary.scalar("Actor_Loss", self.actor_loss)
        tf.summary.scalar("Critic_Loss", self.critic_loss)
        tf.summary.scalar("Recon_Loss", self.r_loss)
        tf.summary.scalar("KL_Loss", self.kl_loss)
        tf.summary.scalar("Task_Loss", self.task_loss)
        tf.summary.scalar("VAE_Loss", self.vae_loss)
        self.summary_op = tf.summary.merge_all()

    def setup_target_network_updates(self):
        actor_init_updates, actor_soft_updates = get_target_updates(self.actor.vars, self.target_actor.vars, self.tau)
        critic_init_updates, critic_soft_updates = get_target_updates(self.critic.vars, self.target_critic.vars, self.tau)
        
        self.target_init_updates = [actor_init_updates, critic_init_updates]
        self.target_soft_updates = [actor_soft_updates, critic_soft_updates]

    def setup_param_noise(self, latent0):
        assert self.param_noise is not None

        # Configure perturbed actor.
        param_noise_actor = copy(self.actor)
        param_noise_actor.name = 'param_noise_actor'
        self.perturbed_actor_tf = param_noise_actor(latent0)
        logger.info('setting up param noise')
        self.perturb_policy_ops = get_perturbed_actor_updates(self.actor, param_noise_actor, self.param_noise_stddev)

        # Configure separate copy for stddev adoption.
        adaptive_param_noise_actor = copy(self.actor)
        adaptive_param_noise_actor.name = 'adaptive_param_noise_actor'
        adaptive_actor_tf = adaptive_param_noise_actor(latent0)
        self.perturb_adaptive_policy_ops = get_perturbed_actor_updates(self.actor, adaptive_param_noise_actor, self.param_noise_stddev)
        self.adaptive_policy_distance = tf.sqrt(tf.reduce_mean(tf.square(self.actor_tf - adaptive_actor_tf)))

    def setup_vae_optimizer(self):
        logger.info('setting up vae optimizer')
        # reconstruction loss
        self.r_loss = tf.reduce_sum(tf.square(self.normalized_obs0 - self.y), reduction_indices = [1,2,3])
        self.r_loss = tf.reduce_mean(self.r_loss)
        # augmented kl loss per dim
        self.kl_loss = - 0.5 * tf.reduce_sum((1 + self.logvar - tf.square(self.mu) - tf.exp(self.logvar)), reduction_indices = 1)
        self.kl_loss = tf.maximum(self.kl_loss, self.kl_tolerance * self.latent_size)
        self.kl_loss = tf.reduce_mean(self.kl_loss)
        # task loss
        self.task_loss = -self.beta * tf.reduce_mean(self.critic_with_actor_tf)
        # Overall VAE loss
        self.vae_loss = self.r_loss + self.kl_loss + self.task_loss
        
        encoder_shapes = [var.get_shape().as_list() for var in self.encoder.trainable_vars]
        encoder_nb_params = sum([reduce(lambda x, y: x * y, shape) for shape in encoder_shapes])
        logger.info('  encoder shapes: {}'.format(encoder_shapes))
        logger.info('  encoder params: {}'.format(encoder_nb_params))
        decoder_shapes = [var.get_shape().as_list() for var in self.decoder.trainable_vars]
        decoder_nb_params = sum([reduce(lambda x, y: x * y, shape) for shape in decoder_shapes])
        logger.info('  decoder shapes: {}'.format(decoder_shapes))
        logger.info('  decoder params: {}'.format(decoder_nb_params))
        vae_trainable_vars = self.encoder.trainable_vars + self.decoder.trainable_vars
        self.vae_grads = U.flatgrad(self.vae_loss, vae_trainable_vars, clip_norm=self.clip_norm)
        self.vae_optimizer = MpiAdam(var_list=vae_trainable_vars, beta1=0.9, beta2=0.999, epsilon=self.epsilon)

    def setup_actor_optimizer(self):
        logger.info('setting up actor optimizer')
        self.actor_loss = -tf.reduce_mean(self.critic_with_actor_tf)
        actor_shapes = [var.get_shape().as_list() for var in self.actor.trainable_vars]
        actor_nb_params = sum([reduce(lambda x, y: x * y, shape) for shape in actor_shapes])
        logger.info('  actor shapes: {}'.format(actor_shapes))
        logger.info('  actor params: {}'.format(actor_nb_params))
        self.actor_grads = U.flatgrad(self.actor_loss, self.actor.trainable_vars, clip_norm=self.clip_norm)
        self.actor_optimizer = MpiAdam(var_list=self.actor.trainable_vars, beta1=0.9, beta2=0.999, epsilon=self.epsilon)

    def setup_critic_optimizer(self):
        logger.info('setting up critic optimizer')
        normalized_critic_target_tf = tf.clip_by_value(normalize(self.critic_target, self.ret_rms), self.return_range[0], self.return_range[1])
        self.critic_loss = tf.reduce_mean(tf.square(self.normalized_critic_tf - normalized_critic_target_tf))
        if self.critic_l2_reg > 0.:
            critic_reg_vars = [var for var in self.critic.trainable_vars if var.name.endswith('/w:0') and 'output' not in var.name]
            for var in critic_reg_vars:
                logger.info('  regularizing: {}'.format(var.name))
            logger.info('  applying l2 regularization with {}'.format(self.critic_l2_reg))
            critic_reg = tc.layers.apply_regularization(
                tc.layers.l2_regularizer(self.critic_l2_reg),
                weights_list=critic_reg_vars
            )
            self.critic_loss += critic_reg
        critic_shapes = [var.get_shape().as_list() for var in self.critic.trainable_vars]
        critic_nb_params = sum([reduce(lambda x, y: x * y, shape) for shape in critic_shapes])
        logger.info('  critic shapes: {}'.format(critic_shapes))
        logger.info('  critic params: {}'.format(critic_nb_params))
        self.critic_grads = U.flatgrad(self.critic_loss, self.critic.trainable_vars, clip_norm=self.clip_norm)
        self.critic_optimizer = MpiAdam(var_list=self.critic.trainable_vars,
            beta1=0.9, beta2=0.999, epsilon=self.epsilon)

    def setup_popart(self):
        # See https://arxiv.org/pdf/1602.07714.pdf for details.
        self.old_std = tf.placeholder(tf.float32, shape=[1], name='old_std')
        new_std = self.ret_rms.std
        self.old_mean = tf.placeholder(tf.float32, shape=[1], name='old_mean')
        new_mean = self.ret_rms.mean

        self.renormalize_Q_outputs_op = []
        for vs in [self.critic.output_vars, self.target_critic.output_vars]:
            assert len(vs) == 2
            M, b = vs
            assert 'kernel' in M.name
            assert 'bias' in b.name
            assert M.get_shape()[-1] == 1
            assert b.get_shape()[-1] == 1
            self.renormalize_Q_outputs_op += [M.assign(M * self.old_std / new_std)]
            self.renormalize_Q_outputs_op += [b.assign((b * self.old_std + self.old_mean - new_mean) / new_std)]

    def setup_stats(self):
        ops = []
        names = []

        if self.normalize_returns:
            ops += [self.ret_rms.mean, self.ret_rms.std]
            names += ['ret_rms_mean', 'ret_rms_std']

        if self.normalize_observations:
            ops += [tf.reduce_mean(self.obs_rms.mean), tf.reduce_mean(self.obs_rms.std)]
            names += ['obs_rms_mean', 'obs_rms_std']

        ops += [tf.reduce_mean(self.critic_tf)]
        names += ['reference_Q_mean']
        ops += [reduce_std(self.critic_tf)]
        names += ['reference_Q_std']

        ops += [tf.reduce_mean(self.critic_with_actor_tf)]
        names += ['reference_actor_Q_mean']
        ops += [reduce_std(self.critic_with_actor_tf)]
        names += ['reference_actor_Q_std']

        ops += [tf.reduce_mean(self.actor_tf)]
        names += ['reference_action_mean']
        ops += [reduce_std(self.actor_tf)]
        names += ['reference_action_std']

        if self.param_noise:
            ops += [tf.reduce_mean(self.perturbed_actor_tf)]
            names += ['reference_perturbed_action_mean']
            ops += [reduce_std(self.perturbed_actor_tf)]
            names += ['reference_perturbed_action_std']

        self.stats_ops = ops
        self.stats_names = names

    def extract_observation(self, obs):
        flat_image, additional = np.split(obs, [-2], axis=1)
        image = np.reshape(flat_image, self.image_shape)
        return image, additional

    def extract_observation_batch(self, obs):
        flat_image, additional = np.split(obs, [-2], axis=1)
        image = np.reshape(flat_image, (self.batch_size,) + self.image_shape)
        return image, additional

    def step(self, obs, apply_noise=True, compute_Q=True):
        image, additional = self.extract_observation(obs)
        epsilon = np.random.randn(additional.shape[0], self.latent_size)
        if self.param_noise is not None and apply_noise:
            actor_tf = self.perturbed_actor_tf
        else:
            actor_tf = self.actor_tf
        feed_dict = {self.obs0: U.adjust_shape(self.obs0, [image]),
                     self.additional0: U.adjust_shape(self.additional0, [additional]),
                     self.epsilon_ph: U.adjust_shape(self.epsilon_ph, [epsilon])}
        if compute_Q:
            action, q = self.sess.run([actor_tf, self.critic_with_actor_tf], feed_dict=feed_dict)
        else:
            action = self.sess.run(actor_tf, feed_dict=feed_dict)
            q = None

        if self.action_noise is not None and apply_noise:
            noise = self.action_noise()
            assert noise.shape == action[0].shape
            action += noise
        action = np.clip(action, self.action_range[0], self.action_range[1])


        return action, q, None, None

    def store_transition(self, obs0, action, reward, obs1, terminal1):
        reward *= self.reward_scale

        B = obs0.shape[0]
        for b in range(B):
            self.memory.append(obs0[b], action[b], reward[b], obs1[b], terminal1[b])
            if self.normalize_observations:
                self.obs_rms.update(np.array([obs0[b]]))

    def train(self):
        # Get a batch.
        batch = self.memory.sample(batch_size=self.batch_size)
        images0, additionals0 = self.extract_observation_batch(batch['obs0'])
        images1, additionals1 = self.extract_observation_batch(batch['obs1'])
        epsilon = np.random.randn(additionals0.shape[0], self.latent_size)

        # Get all gradients and perform a synced update.
        ops = [self.actor_grads, self.actor_loss, self.critic_grads, self.critic_loss, self.vae_grads, self.vae_loss, self.summary_op]
        actor_grads, actor_loss, critic_grads, critic_loss, vae_grads, vae_loss, summary = self.sess.run(ops, feed_dict={
            self.obs1: images1,
            self.additional1: additionals1,
            self.rewards: batch['rewards'],
            self.terminals1: batch['terminals1'].astype('float32'),
            self.obs0: images0,
            self.additional0: additionals0,
            self.epsilon_ph: epsilon,
            self.actions: batch['actions']
        })
        self.actor_optimizer.update(actor_grads, stepsize=self.actor_lr)
        self.critic_optimizer.update(critic_grads, stepsize=self.critic_lr)
        self.vae_optimizer.update(vae_grads, stepsize=self.actor_lr)

        return critic_loss, actor_loss, summary

    def initialize(self, sess):
        self.sess = sess
        self.sess.run(tf.global_variables_initializer())
        self.actor_optimizer.sync()
        self.critic_optimizer.sync()
        self.sess.run(self.target_init_updates)

    def update_target_net(self):
        self.sess.run(self.target_soft_updates)

    def get_stats(self):
        if self.stats_sample is None:
            # Get a sample and keep that fixed for all further computations.
            # This allows us to estimate the change in value for the same set of inputs.
            self.stats_sample = self.memory.sample(batch_size=self.batch_size)            
            images0, additionals0 = self.extract_observation_batch(self.stats_sample['obs0'])
            self.stats_sample['image0'] = images0
            self.stats_sample['additional0'] = additionals0

        epsilon = np.random.randn(self.stats_sample['additional0'].shape[0], self.latent_size)
        values = self.sess.run(self.stats_ops, feed_dict={
            self.obs0: self.stats_sample['image0'],
            self.additional0: self.stats_sample['additional0'],
            self.actions: self.stats_sample['actions'],
            self.epsilon_ph: epsilon
        })

        names = self.stats_names[:]
        assert len(names) == len(values)
        stats = dict(zip(names, values))

        if self.param_noise is not None:
            stats = {**stats, **self.param_noise.get_stats()}

        return stats

    def adapt_param_noise(self):
        try:
            from mpi4py import MPI
        except ImportError:
            MPI = None

        if self.param_noise is None:
            return 0.

        # Perturb a separate copy of the policy to adjust the scale for the next "real" perturbation.
        batch = self.memory.sample(batch_size=self.batch_size)
        images0, additionals0 = self.extract_observation_batch(batch['obs0'])
        epsilon = np.random.randn(additionals0.shape[0], self.latent_size)
        self.sess.run(self.perturb_adaptive_policy_ops, feed_dict={
            self.param_noise_stddev: self.param_noise.current_stddev,
        })
        distance = self.sess.run(self.adaptive_policy_distance, feed_dict={
            self.obs0: images0,
            self.additional0: additionals0,
            self.param_noise_stddev: self.param_noise.current_stddev,
            self.epsilon_ph: epsilon
        })

        if MPI is not None:
            mean_distance = MPI.COMM_WORLD.allreduce(distance, op=MPI.SUM) / MPI.COMM_WORLD.Get_size()
        else:
            mean_distance = distance

        if MPI is not None:
            mean_distance = MPI.COMM_WORLD.allreduce(distance, op=MPI.SUM) / MPI.COMM_WORLD.Get_size()
        else:
            mean_distance = distance

        self.param_noise.adapt(mean_distance)
        return mean_distance

    def reset(self):
        # Reset internal state after an episode is complete.
        if self.action_noise is not None:
            self.action_noise.reset()
        if self.param_noise is not None:
            self.sess.run(self.perturb_policy_ops, feed_dict={
                self.param_noise_stddev: self.param_noise.current_stddev,
            })
            
    def save(self, path):
        save_variables(path)
            
    def load(self, path):
        load_variables(path)
