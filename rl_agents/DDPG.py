# Library Imports
from copy import copy
import numpy as np
from numpy.lib import utils
import tensorflow as tf
from collections import deque
import random
from replay_buffers.PER import PrioritizedReplayBuffer
from replay_buffers.utils import LinearSchedule
import gc
tf.random.set_seed(0)
np.random.seed(0)


class OUNoise:
    def __init__(self, action_dimension, scale=0.1, mu=0, theta=0.15, sigma=0.2):
        self.action_dimension = action_dimension
        self.scale = scale
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.action_dimension) * self.mu
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dimension) * self.mu

    def __call__(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state * self.scale


class Critic(tf.keras.Model):
    """Defines a Critic Deep Learning Network"""

    def __init__(self, density=512, name='critic'):
        super(Critic, self).__init__()

        self.model_name = name
        self.checkpoint = self.model_name + '.h5'

        self.H1 = tf.keras.layers.Dense(density, activation='relu')
        self.H2 = tf.keras.layers.Dense(density, activation='relu')
        self.drop = tf.keras.layers.Dropout(0.1)
        self.H3 = tf.keras.layers.Dense(density, activation='relu')
        self.H4 = tf.keras.layers.Dense(density, activation='relu')
        self.Q = tf.keras.layers.Dense(1, activation=None)

    @tf.function()
    def call(self, state, action):
        state = tf.cast(state, tf.float64)
        action = tf.cast(action, tf.float64)
        action = self.H1(tf.concat([state, action], axis=1))
        action = self.H2(action)
        action = self.drop(action)
        action = self.H3(action)
        action = self.H4(action)
        Q = self.Q(action)
        return Q


class Actor(tf.keras.Model):
    """Defines a Actor Deep Learning Network"""

    def __init__(self, n_actions, density=512, name='actor'):
        super(Actor, self).__init__()

        self.model_name = name
        self.checkpoint = self.model_name + '.h5'

        self.H1 = tf.keras.layers.Dense(density, activation='relu')
        self.H2 = tf.keras.layers.Dense(density, activation='relu')
        self.drop = tf.keras.layers.Dropout(0.1)
        self.H3 = tf.keras.layers.Dense(density, activation='relu')
        self.H4 = tf.keras.layers.Dense(density, activation='relu')
        self.mu = tf.keras.layers.Dense(n_actions, activation='tanh')

    @tf.function()
    def call(self, state):
        state = self.H1(state)
        state = self.H2(state)
        state = self.drop(state)
        state = self.H3(state)
        state = self.H4(state)
        mu = self.mu(state)
        return mu


class Agent:
    def __init__(self, env, datapath, n_games, alpha=0.0001,
                 beta=0.001, gamma=0.99, tau=0.005, batch_size=64,
                 noise='normal', per_alpha=0.6, per_beta=0.4):

        self.env = env
        self.gamma = tf.convert_to_tensor([gamma], dtype=tf.float32)
        self.tau = tf.convert_to_tensor([tau], dtype=tf.float32)
        self.n_actions = env.action_space.shape[0]
        self.obs_shape = env.observation_space.shape[0]
        self.datapath = datapath
        self.n_games = n_games
        self.optim_steps = 0
        self.max_size = int(env.max_episode_length * n_games)
        self.memory = PrioritizedReplayBuffer(self.max_size, per_alpha)
        self.beta_scheduler = LinearSchedule(n_games, per_beta, 0.99)

        self.batch_size = batch_size
        self.noise = noise
        self.max_action = env.action_space.high
        self.min_action = env.action_space.low

        self.actor = Actor(self.n_actions, name='actor')
        self.critic = Critic(name='critic')
        self.target_actor = Actor(self.n_actions, name='target_actor')
        self.target_critic = Critic(name='target_critic')

        self.actor.compile(tf.keras.optimizers.Adam(alpha))
        self.critic.compile(tf.keras.optimizers.Adam(beta))
        self.target_actor.compile(tf.keras.optimizers.Adam(alpha))
        self.target_critic.compile(tf.keras.optimizers.Adam(beta))

        if self.noise == 'normal':
            self.noise_param = 0.1
        elif self.noise == 'ou':
            self.noise = OUNoise(self.n_actions)
        elif self.noise == 'param':
            self.distances = []
            self.scalar = 0.01
            self.scalar_decay = 0.1
            self.desired_distance = 0.1
            self.noisy_actor = Actor(self.n_actions, name='noisy_actor')
            # Fire-up 'noisy_actor' to set params.
            obs = env.observation_space.sample()
            state = tf.convert_to_tensor([obs], dtype=tf.float64)
            self.noisy_actor(state)

        self.update_networks()

    def update_noisy_actor(self):
        weights = []
        for weight in self.actor.weights:
            noise = tf.random.normal(
                shape=weight.shape, stddev=self.scalar)
            weights.append(weight + noise)
        self.noisy_actor.set_weights(weights)

    def update_networks(self):
        tau = self.tau

        weights = []
        targets = self.target_actor.weights
        for i, weight in enumerate(self.actor.weights):
            weights.append(weight * tau + targets[i] * (1 - tau))
        self.target_actor.set_weights(weights)

        weights = []
        targets = self.target_critic.weights
        for i, weight in enumerate(self.critic.weights):
            weights.append(weight * tau + targets[i] * (1 - tau))
        self.target_critic.set_weights(weights)

    def store(self, state, action, reward, new_state, done):
        self.memory.add(state, action, reward, new_state, done)

    def save_models(self):
        self.actor.save_weights(self.datapath + self.actor.checkpoint)
        self.critic.save_weights(self.datapath + self.critic.checkpoint)
        self.target_actor.save_weights(self.datapath +
                                       self.target_actor.checkpoint)
        self.target_critic.save_weights(self.datapath +
                                        self.target_critic.checkpoint)

    def load_models(self):
        self.actor.load_weights(self.datapath + self.actor.checkpoint)
        self.critic.load_weights(self.datapath + self.critic.checkpoint)
        self.target_actor.load_weights(self.datapath +
                                       self.target_actor.checkpoint)
        self.target_critic.load_weights(self.datapath +
                                        self.target_critic.checkpoint)

    def choose_action(self, observation):
        state = tf.convert_to_tensor([observation], dtype=tf.float64)
        action = self.actor(state)

        if self.noise == 'normal':
            action += tf.random.normal([self.n_actions], 0.0, self.noise_param)

        elif self.noise == 'ou':
            action += self.noise()

        elif self.noise == 'param':
            self.update_noisy_actor()
            action_noised = self.noisy_actor(state)
            distance = tf.linalg.norm(action - action_noised)
            self.distances.append(distance)
            if distance > self.desired_distance:
                self.scalar *= self.scalar_decay
            if distance < self.desired_distance:
                self.scalar /= self.scalar_decay
            action += action_noised

        action = tf.clip_by_value(action, self.min_action, self.max_action)
        return action[0].numpy()

    def optimize(self):
        if len(self.memory._storage) < self.batch_size:
            return

        beta = self.beta_scheduler.value(self.optim_steps)
        state, action, reward, new_state, done, weights, indices = self.memory.sample(
            self.batch_size, beta)

        state = tf.convert_to_tensor(np.vstack(state), dtype=tf.float64)
        action = tf.convert_to_tensor(np.vstack(action), dtype=tf.float64)
        done = tf.convert_to_tensor(np.vstack(1 - done), dtype=tf.float32)
        reward = tf.convert_to_tensor(np.vstack(reward), dtype=tf.float32)
        weights = tf.convert_to_tensor(
            np.sqrt(np.vstack(weights)), dtype=tf.float32)
        new_state = tf.convert_to_tensor(
            np.vstack(new_state), dtype=tf.float64)

        with tf.GradientTape() as tape:
            # Compute the Q value estimate of the target network
            Q_target = self.target_critic(
                new_state, self.target_actor(new_state))
            # Compute Y
            Y = reward + (done * [self.gamma] * Q_target)
            # Compute Q value estimate of critic
            Q = self.critic(state, action)
            # Calculate TD errors
            TD_errors = (Y - Q)
            # Weight TD errors
            weighted_TD_errors = TD_errors * weights
            # Create a zero tensor
            zero_tensor = tf.zeros(weighted_TD_errors.shape)
            # Compute critic loss, MSE of weighted TD_r
            critic_loss = tf.keras.losses.mse(weighted_TD_errors, zero_tensor)

        critic_network_gradient = tape.gradient(
            critic_loss, self.critic.trainable_variables)
        self.critic.optimizer.apply_gradients(zip(
            critic_network_gradient, self.critic.trainable_variables))

        with tf.GradientTape() as tape:
            actor_loss = - \
                tf.math.reduce_mean(self.critic(state, self.actor(state)))

        actor_network_gradient = tape.gradient(
            actor_loss, self.actor.trainable_variables)
        self.actor.optimizer.apply_gradients(zip(
            actor_network_gradient, self.actor.trainable_variables))

        td_errors = TD_errors.numpy()
        new_priorities = np.abs(td_errors) + 1e-6
        self.memory.update_priorities(indices, new_priorities)

        self.update_networks()

        self.optim_steps += 1.0
