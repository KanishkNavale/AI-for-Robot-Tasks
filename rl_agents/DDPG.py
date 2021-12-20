# Library Imports
import numpy as np
import tensorflow as tf
from collections import deque
import random
tf.random.set_seed(0)
np.random.seed(0)


class OUNoise:
    def __init__(self, action_dimension, mu=0, theta=0.015, sigma=0.03):
        self.action_dimension = action_dimension
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
        return self.state


class PrioritizedReplayBuffer():
    def __init__(self, maxlen, alpha, beta):
        self.buffer = deque(maxlen=maxlen)
        self.priorities = deque(maxlen=maxlen)
        self.alpha = alpha
        self.beta = beta

    def store_transition(self, state, action, reward, new_state, done):
        self.buffer.append((state, action, reward, new_state, done))
        self.priorities.append(max(self.priorities, default=1))
        self.mem_cntr = len(self.buffer)

    def get_probabilities(self):
        scaled_priorities = np.array(self.priorities) ** self.alpha
        sample_probabilities = scaled_priorities / sum(scaled_priorities)
        return sample_probabilities

    def get_weights(self, probabilities):
        weights = np.power(len(self.buffer), -self.beta) * \
            np.power(probabilities, -self.beta)
        weights_normalized = weights / max(weights)
        return weights_normalized

    def sample(self, batch_size):
        sample_size = min(len(self.buffer), batch_size)
        sample_probs = self.get_probabilities()
        sample_indices = random.choices(
            range(len(self.buffer)), k=sample_size, weights=sample_probs)
        samples = np.array(self.buffer)[sample_indices]
        weights = self.get_weights(sample_probs[sample_indices])
        return map(list, zip(*samples)), weights, sample_indices

    def set_priorities(self, indices, errors, offset=1e-6):
        for i, e in zip(indices, errors):
            self.priorities[i] = abs(e) + offset


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
    def __init__(self, env, datapath, alpha=0.0001,
                 beta=0.001, gamma=0.99, max_size=250000, tau=0.005,
                 batch_size=32, per_alpha=0.7, per_beta=0.5):

        self.env = env
        self.gamma = gamma
        self.tau = tau
        self.n_actions = env.action_space.shape[0]
        self.obs_shape = env.observation_space.shape[0]
        self.datapath = datapath
        self.per_beta = per_beta
        self.memory = PrioritizedReplayBuffer(max_size, per_alpha, per_beta)

        self.batch_size = batch_size
        self.noise = OUNoise(self.n_actions)
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
        self.update_networks()

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
        self.memory.store_transition(state, action, reward, new_state, done)

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
        state = tf.convert_to_tensor([observation], dtype=tf.float32)
        actions = self.actor(state)
        actions += self.noise()
        actions = tf.clip_by_value(actions, self.min_action, self.max_action)
        return actions[0].numpy()

    def optimize(self):
        if self.memory.mem_cntr < self.batch_size:
            return

        experience, weights, indices = self.memory.sample(self.batch_size)
        state, action, reward, new_state, done = experience

        state = np.vstack(state)
        new_state = np.vstack(new_state)
        action = np.vstack(action)
        done = np.array(done)
        reward = np.array(reward)

        with tf.GradientTape() as tape:
            _mui = self.target_actor(new_state)
            _Q = tf.squeeze(self.target_critic(new_state, _mui), 1)
            Q = tf.squeeze(self.critic(state, action), 1)
            TD_error = reward + (self.gamma * _Q) * (1.0 - done)
            critic_loss = tf.keras.losses.mse(
                weights * tf.math.square(TD_error), Q)

        critic_network_gradient = tape.gradient(
            critic_loss, self.critic.trainable_variables)
        self.critic.optimizer.apply_gradients(zip(
            critic_network_gradient, self.critic.trainable_variables))

        with tf.GradientTape() as tape:
            new_policy_actions = self.actor(state)
            actor_loss = -self.critic(state, new_policy_actions)
            actor_loss = tf.math.reduce_mean(actor_loss)

        actor_network_gradient = tape.gradient(
            actor_loss, self.actor.trainable_variables)
        self.actor.optimizer.apply_gradients(zip(
            actor_network_gradient, self.actor.trainable_variables))

        self.memory.set_priorities(indices, TD_error.numpy())

        self.update_networks()
