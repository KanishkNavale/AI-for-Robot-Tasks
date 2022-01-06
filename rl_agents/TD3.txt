# Library Imports
import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from replay_buffers.PER import PrioritizedReplayBuffer
from replay_buffers.utils import LinearSchedule
import copy
T.manual_seed(0)
np.random.seed(0)


device = T.device("cuda:0" if T.cuda.is_available() else "cpu")


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


class Critic(nn.Module):
    """Defines a Critic Deep Learning Network"""

    def __init__(self, input_dim, beta, density=512, name='critic'):
        super(Critic, self).__init__()

        self.model_name = name
        self.checkpoint = self.model_name

        # Architecture
        self.H1 = nn.Linear(input_dim, density)
        self.H2 = nn.Linear(density, density)
        self.drop = nn.Dropout(p=0.1)
        self.H3 = nn.Linear(density, density)
        self.H4 = nn.Linear(density, density)
        self.Q = nn.Linear(density, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=beta)
        self.to(device)

    def forward(self, state, action):
        value = T.hstack((state, action))
        value = F.relu(self.H1(value))
        value = F.relu(self.H2(value))
        value = self.drop(value)
        value = F.relu(self.H3(value))
        value = F.relu(self.H4(value))
        value = self.Q(value)
        return value

    def save_model(self, path):
        T.save(self.state_dict(), path + self.checkpoint)

    def load_model(self, path):
        self.load_state_dict(T.load(path + self.checkpoint))


class Actor(nn.Module):
    """Defines a Actor Deep Learning Network"""

    def __init__(self, input_dim, n_actions, alpha, density=512, name='actor'):
        super(Actor, self).__init__()

        self.model_name = name
        self.checkpoint = self.model_name

        # Architecture
        self.H1 = nn.Linear(input_dim, density)
        self.H2 = nn.Linear(density, density)
        self.drop = nn.Dropout(p=0.1)
        self.H3 = nn.Linear(density, density)
        self.H4 = nn.Linear(density, density)
        self.mu = nn.Linear(density, n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.to(device)

    def forward(self, state):
        action = F.relu(self.H1(state))
        action = F.relu(self.H2(action))
        action = self.drop(action)
        action = F.relu(self.H3(action))
        action = F.relu(self.H4(action))
        action = T.tanh(self.mu(action))
        return action

    def save_model(self, path):
        T.save(self.state_dict(), path + self.checkpoint)

    def load_model(self, path):
        self.load_state_dict(T.load(path + self.checkpoint))


class Agent():
    def __init__(self, env, datapath, n_games, alpha=0.0001,
                 beta=0.001, gamma=0.99, tau=0.005, batch_size=64,
                 noise='normal', per_alpha=0.6, per_beta=0.4, enable_HER=False):

        self.env = env
        self.gamma = T.tensor(gamma, dtype=T.float32).to(device)
        self.tau = T.tensor(tau, dtype=T.float32).to(device)
        self.n_actions = env.action_space.shape[0]
        self.obs_shape = env.observation_space.shape[0]
        self.datapath = datapath
        self.n_games = n_games
        self.optim_steps = 0
        self.max_size = 2500000
        self.memory = PrioritizedReplayBuffer(self.max_size, per_alpha)
        self.beta_scheduler = LinearSchedule(n_games, per_beta, 0.99)
        self.her = enable_HER

        self.batch_size = batch_size
        self.noise = noise
        self.max_action = T.tensor(
            env.action_space.high, dtype=T.float32).to(device)
        self.min_action = T.tensor(
            env.action_space.low, dtype=T.float32).to(device)

        self.actor = Actor(self.obs_shape, self.n_actions, alpha, name='actor')
        self.target_actor = Actor(
            self.obs_shape, self.n_actions, alpha, name='target_actor')

        self.critic_1 = Critic(
            self.obs_shape + self.n_actions, beta, name='critic_1')
        self.critic_2 = Critic(
            self.obs_shape + self.n_actions, beta, name='critic_2')
        self.target_critic_1 = Critic(
            self.obs_shape + self.n_actions, beta, name='target_critic_1')
        self.target_critic_2 = Critic(
            self.obs_shape + self.n_actions, beta, name='target_critic_2')

        if self.noise == 'normal':
            self.noise_param = 0.001

        elif self.noise == 'ou':
            self.noise = OUNoise(self.n_actions)

        elif self.noise == 'param':
            self.distances = []
            self.scalar = 0.001
            self.scalar_decay = 0.01
            self.desired_distance = 0.01
            self.noisy_actor = Actor(
                self.obs_shape, self.n_actions, alpha, name='noisy_actor')

        else:
            raise NotImplementedError("This noise is not implmented!")

        self.update_network()

    def save_models(self):
        self.actor.save_model(self.datapath)
        self.target_actor.save_model(self.datapath)
        self.critic_1.save_model(self.datapath)
        self.target_critic_1.save_model(self.datapath)
        self.critic_2.save_model(self.datapath)
        self.target_critic_2.save_model(self.datapath)

    def load_models(self):
        self.actor.load_model(self.datapath)
        self.target_actor.load_model(self.datapath)
        self.critic_1.load_model(self.datapath)
        self.target_critic_1.load_model(self.datapath)
        self.critic_2.load_model(self.datapath)
        self.target_critic_2.load_model(self.datapath)

    def update_network(self):
        """
        Description,
            Performs network updates as per TD3 Policy
        """
        tau = self.tau

        for critic_weights, target_critic_weights in zip(self.critic_1.parameters(), self.target_critic_1.parameters()):
            target_critic_weights.data.copy_(
                tau * critic_weights.data + (1 - tau) * target_critic_weights.data)

        for critic_weights, target_critic_weights in zip(self.critic_2.parameters(), self.target_critic_2.parameters()):
            target_critic_weights.data.copy_(
                tau * critic_weights.data + (1 - tau) * target_critic_weights.data)

        for actor_weights, target_actor_weights in zip(self.actor.parameters(), self.target_actor.parameters()):
            target_actor_weights.data.copy_(
                tau * actor_weights.data + (1 - tau) * target_actor_weights.data)

    def choose_action(self, observation):
        self.actor.eval()
        state = T.tensor(observation, dtype=T.float32).to(device)
        action = self.actor.forward(state)

        if self.noise == 'normal':
            noise = np.random.uniform(0.0, self.noise_param, action.shape[0])
            action += T.tensor(noise, dtype=T.float32).to(device)

        elif self.noise == 'ou':
            action += T.tensor(self.noise(), dtype=T.float32).to(device)

        elif self.noise == 'param':
            self.update_noisy_actor()
            self.noisy_actor.eval()
            action_noised = self.noisy_actor(state)
            distance = T.linalg.norm(action - action_noised)
            self.distances.append(distance)
            if distance > self.desired_distance:
                self.scalar *= self.scalar_decay
            if distance < self.desired_distance:
                self.scalar /= self.scalar_decay
            action += action_noised

        action = T.clamp(action, self.min_action, self.max_action)
        self.actor.train()
        return action.detach().cpu().numpy()

    def store(self, states, actions, rewards, next_states, dones, targets):
        if self.her:
            # Hyperparameter for Future Goal Sampling
            k = 4

            # Augment the replay buffer
            T = len(actions)

            for index in range(T):
                for _ in range(k):
                    # Always fetch index of upcoming episode transitions
                    future = np.random.randint(index, T)

                    # Unpack the buffers using the future index
                    future_goal = next_states[future]
                    HER_goal = copy.deepcopy(future_goal)

                    # Compute HER Reward
                    reward, done = self.env.compute_reward(
                        HER_goal, future_goal)

                    # Repack augmented episode transitions
                    obs = states[future]
                    state = np.concatenate((obs, HER_goal))

                    next_obs = next_states[future]
                    next_state = np.concatenate((next_obs, HER_goal))

                    action = actions[future]

                    # Add augmented episode transitions to agent's memory
                    self.memory.add(state, action, reward, next_state, done)

        else:
            for state, action, reward, new_state, done, target in zip(states, actions, rewards, next_states, dones, targets):
                state = np.concatenate((state, target))
                new_state = np.concatenate((new_state, target))
                self.memory.add(state, action, reward, new_state, done)

    def optimize(self):
        """
        Description,
            Performs TD3 policy optimization step.
        """
        if len(self.memory._storage) < self.batch_size:
            return

        beta = self.beta_scheduler.value(self.optim_steps)
        state, action, reward, new_state, done, weights, indices = self.memory.sample(
            self.batch_size, beta)

        state = T.tensor(np.vstack(state), dtype=T.float32).to(device)
        action = T.tensor(np.vstack(action), dtype=T.float32).to(device)
        done = T.tensor(np.vstack(done), dtype=T.bool).to(device)
        reward = T.tensor(np.vstack(reward), dtype=T.float32).to(device)
        new_state = T.tensor(np.vstack(new_state), dtype=T.float32).to(device)
        weights = T.tensor(np.vstack(weights), dtype=T.float32).to(device)

        self.target_actor.eval()
        self.target_critic_1.eval()
        self.critic_1.eval()
        self.target_critic_2.eval()
        self.critic_2.eval()

        target_actions = self.target_actor.forward(new_state)

        target_q1 = self.target_critic_1.forward(new_state, target_actions)
        target_q2 = self.target_critic_2.forward(new_state, target_actions)

        target_q1[done] = 0.0
        target_q2[done] = 0.0

        q1 = self.critic_1.forward(state, action)
        q2 = self.critic_2.forward(state, action)

        q1 = q1.squeeze()
        q2 = q2.squeeze()
        target_q1 = target_q1.squeeze()
        target_q2 = target_q2.squeeze()
        reward = reward.squeeze()

        critic_value = T.min(target_q1, target_q2)
        target = reward + self.gamma * critic_value

        self.critic_1.optimizer.zero_grad()
        self.critic_2.optimizer.zero_grad()

        q1_loss = F.mse_loss(target, q1)
        q2_loss = F.mse_loss(target, q2)
        TD_errors = 0.5 * (q1_loss + q2_loss)

        # Weight TD errors
        weighted_TD_errors = T.mul(TD_errors, weights)
        zero_tensor = T.zeros_like(
            weighted_TD_errors, dtype=T.float32).to(device)

        # Compute & Update Critic losses
        critic_loss = F.mse_loss(weighted_TD_errors, zero_tensor)

        critic_loss.backward()
        self.critic_1.optimizer.step()
        self.critic_2.optimizer.step()

        td_errors = target.detach().cpu().numpy()
        new_priorities = np.abs(td_errors) + 1e-6
        self.memory.update_priorities(indices, new_priorities)

        self.optim_steps += 1.0

        if self.optim_steps % 2 != 0:
            return

        self.actor.optimizer.zero_grad()
        actor_q1_loss = self.critic_1.forward(state, self.actor.forward(state))
        actor_loss = -T.mean(actor_q1_loss)
        actor_loss.backward()
        self.actor.optimizer.step()

        self.update_network()
