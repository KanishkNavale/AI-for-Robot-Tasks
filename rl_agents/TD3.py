import numpy as np
import torch as T
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import os


class ReplayBuffer:
    """Defines the Buffer dataset from which the agent learns"""
    def __init__(self, max_size, input_shape, dim_actions):
        """
        Description,
        Initializes matrices as dataframes.
        Args:
            max_size ([int]): Max Size of the Buffer
            input_shape ([type]): Observation Shape
            dim_actions ([type]): Dimension of action
        """
        self.mem_size = max_size
        self.mem_cntr = 0
        self.state_memory = np.zeros((self.mem_size, input_shape),
                                     dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, input_shape),
                                         dtype=np.float32)
        self.action_memory = np.zeros((self.mem_size, dim_actions),
                                      dtype=np.float32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool)

    def store_transition(self, state, action, reward, new_state, done):
        """
        Description,
            Adds experience to it's w.r.t matrices dataframes.
        Args:
            state ([np.array]): State
            action ([int]): action
            reward ([np.float32]): reward
            new_state ([np.array]): State
            done ([bool]): Done Flag
        """
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = new_state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done
        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        """
        Description,
            Samples random batch of experiences from dataframe.
        Args:
            batch_size ([int]): No. of Experiences.
        Returns:
            states ([np.array]): State
            actions ([int]): action
            rewards ([np.float32]): reward
            new_states ([np.array]): State
            dones ([bool]): Done Flag
        """
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size, replace=False)
        states = self.state_memory[batch]
        _states = self.new_state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        dones = self.terminal_memory[batch]
        return states, actions, rewards, _states, dones


class CriticNetwork(nn.Module):
    def __init__(self, beta, input_dims, fc1_dims, fc2_dims, n_actions,
                 name, chkpt_dir):
        """
        Description,
            Initializes Critic Network.
        Args:
            beta ([np.float32]): learning rate.
            input_dims ([int]): state shape.
            fc1_dims ([int]): Hidden Layer 1 dimension.
            fc2_dims ([int]): Hidden Layer 2 dimension.
            n_actions ([int]): action shape.
            name ([str]): Name of the Network
            chkpt_dir (str, optional): Data Directory. Defaults to 'data/'.
        """
        super(CriticNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'_td3')

        self.fc1 = nn.Linear(self.input_dims + n_actions, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.q1 = nn.Linear(self.fc2_dims, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=beta)
        self.device = T.device('cpu')

        self.to(self.device)

    def forward(self, state, action):
        """
        Description,
            Solves the forward Propogation as per graph.
        Args:
            states ([np.array]): State
            actions ([int]): action
        Returns:
            [np.float32]: Value.
        """
        q1_action_value = self.fc1(T.cat([state, action], dim=1))
        q1_action_value = F.relu(q1_action_value)
        q1_action_value = self.fc2(q1_action_value)
        q1_action_value = F.relu(q1_action_value)

        q1 = self.q1(q1_action_value)

        return q1

    def save_checkpoint(self):
        """
        Description,
            Saves the model to the checkpoint directory.
        """
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        """
        Description,
            Loads the model from checkpoint directory.
        """
        self.load_state_dict(T.load(self.checkpoint_file))


class ActorNetwork(nn.Module):
    def __init__(self, alpha, input_dims, fc1_dims, fc2_dims,
                 n_actions, name, chkpt_dir):
        """
        Description,
            Initializes Actor Network.
        Args:
            alpha ([np.float32]): learning rate.
            input_dims ([int]): state shape.
            fc1_dims ([int]): Hidden Layer 1 dimension.
            fc2_dims ([int]): Hidden Layer 2 dimension.
            n_actions ([int]): action shape.
            name ([str]): Name of the Network
            chkpt_dir (str, optional): Data Directory. Defaults to 'data/'.
        """
        super(ActorNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'_td3')

        self.fc1 = nn.Linear(self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.mu = nn.Linear(self.fc2_dims, self.n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cpu')

        self.to(self.device)

    def forward(self, state):
        """
        Description,
            Solves the forward Propogation as per graph.
        Args:
            states ([np.array]): State
        Returns:
            [np.float32]: force.
        """
        prob = self.fc1(state)
        prob = F.relu(prob)
        prob = self.fc2(prob)
        prob = F.relu(prob)

        mu = T.tanh(self.mu(prob))

        return mu

    def save_checkpoint(self):
        """
        Description,
            Saves the model to the checkpoint directory.
        """
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        """
        Description,
            Loads the model from checkpoint directory.
        """
        self.load_state_dict(T.load(self.checkpoint_file))


class Agent():
    def __init__(self, obs_shape, n_actions, datapath, alpha=0.0001,
                 beta=0.0002, tau=0.005, gamma=0.99, update_actor_interval=2,
                 max_size=250000, layer1_size=1024,
                 layer2_size=512, batch_size=64, noise=0.1):
        """
        Description,
            Initializes  Agent.
        Args:
            alpha ([np.float32]): Learning rate of Actor.
            beta ([np.float32]): Learning rate of Critic.
            tau ([tau]): Weight transport rate.
            env ([object]): Environment.
            gamma (float, optional): Agent hyperparameter.
            update_actor_interval (int, optional): Agent update Interval.
            n_actions (int, optional): No. of actions.
            max_size (int, optional): Size of Buffer.
            layer1_size (int, optional): Hidden layer 1 Size.
            layer2_size (int, optional): Hidden layer 2 Size.
            batch_size (int, optional): Batch Size for optimization.
            noise (float, optional): Noise. Defaults to 0.1.
        """
        self.gamma = gamma
        self.tau = tau
        self.max_action = 100
        self.min_action = -100

        self.batch_size = batch_size
        self.learn_step_cntr = 0
        self.time_step = 0
        self.n_actions = n_actions
        self.update_actor_iter = update_actor_interval
        self.input_dims = obs_shape
        self.memory = ReplayBuffer(max_size, self.input_dims, n_actions)

        self.actor = ActorNetwork(alpha, self.input_dims, layer1_size,
                                  layer2_size, n_actions=n_actions,
                                  name='actor', chkpt_dir=datapath)

        self.critic_1 = CriticNetwork(beta, self.input_dims, layer1_size,
                                      layer2_size, n_actions=n_actions,
                                      name='critic_1', chkpt_dir=datapath)

        self.critic_2 = CriticNetwork(beta, self.input_dims, layer1_size,
                                      layer2_size, n_actions=n_actions,
                                      name='critic_2', chkpt_dir=datapath)

        self.target_actor = ActorNetwork(alpha, self.input_dims, layer1_size,
                                         layer2_size, n_actions=n_actions,
                                         name='target_actor',
                                         chkpt_dir=datapath)

        self.target_critic_1 = CriticNetwork(beta, self.input_dims,
                                             layer1_size,
                                             layer2_size, n_actions=n_actions,
                                             name='target_critic_1',
                                             chkpt_dir=datapath)

        self.target_critic_2 = CriticNetwork(beta, self.input_dims,
                                             layer1_size,
                                             layer2_size, n_actions=n_actions,
                                             name='target_critic_2',
                                             chkpt_dir=datapath)

        self.noise = noise
        self.update_network_parameters(tau=1)

    def choose_action(self, observation):
        """
        Description,
            Computes action.
        Args:
            observation ([np.float32]): state
        Returns:
            [np.float]: optimal force.
        """
        state = T.tensor(observation, dtype=T.float).to(self.actor.device)
        mu = self.actor.forward(state).to(self.actor.device)
        mu_prime = mu + T.tensor(np.random.normal(scale=self.noise),
                                 dtype=T.float).to(self.actor.device)

        mu_prime = T.clamp(mu_prime, self.min_action, self.max_action)
        self.time_step += 1

        return mu_prime.cpu().detach().numpy()[0]

    def remember(self, state, action, reward, new_state, done):
        """
        Description,
            Adds experience to it's w.r.t matrices dataframes.
        Args:
            state ([np.array]): State
            action ([int]): action
            reward ([np.float32]): reward
            new_state ([np.array]): State
            done ([bool]): Done Flag
        """
        self.memory.store_transition(state, action, reward, new_state, done)

    def optimize(self):
        """
        Description,
            Performs TD3 policy optimization step.
        """
        if self.memory.mem_cntr < self.batch_size:
            return

        state, action, reward, new_state, done = \
            self.memory.sample_buffer(self.batch_size)

        reward = T.tensor(reward, dtype=T.float).to(self.critic_1.device)
        done = T.tensor(done).to(self.critic_1.device)
        state_ = T.tensor(new_state, dtype=T.float).to(self.critic_1.device)
        state = T.tensor(state, dtype=T.float).to(self.critic_1.device)
        action = T.tensor(action, dtype=T.float).to(self.critic_1.device)

        target_actions = self.target_actor.forward(state_)
        target_actions = target_actions + T.clamp(
                         T.tensor(np.random.normal(scale=0.2)), -0.5, 0.5)
        target_actions = T.clamp(target_actions, self.min_action,
                                 self.max_action)

        q1_ = self.target_critic_1.forward(state_, target_actions)
        q2_ = self.target_critic_2.forward(state_, target_actions)

        q1 = self.critic_1.forward(state, action)
        q2 = self.critic_2.forward(state, action)

        q1_[done] = 0.0
        q2_[done] = 0.0

        q1_ = q1_.view(-1)
        q2_ = q2_.view(-1)

        critic_value_ = T.min(q1_, q2_)

        target = reward + self.gamma*critic_value_
        target = target.view(self.batch_size, 1)

        self.critic_1.optimizer.zero_grad()
        self.critic_2.optimizer.zero_grad()

        q1_loss = F.mse_loss(target, q1)
        q2_loss = F.mse_loss(target, q2)
        critic_loss = 0.5 * (q1_loss + q2_loss)
        critic_loss.backward()
        self.critic_1.optimizer.step()
        self.critic_2.optimizer.step()

        self.learn_step_cntr += 1

        if self.learn_step_cntr % self.update_actor_iter != 0:
            return

        self.actor.optimizer.zero_grad()
        actor_q1_loss = self.critic_1.forward(state, self.actor.forward(state))
        actor_loss = -T.mean(actor_q1_loss)
        actor_loss.backward()
        self.actor.optimizer.step()

        self.update_network_parameters()

    def update_network_parameters(self, tau=None):
        """
        Description,
            Performs network updates as per TD3 Policy
        """
        if tau is None:
            tau = self.tau

        actor_params = self.actor.named_parameters()
        critic_1_params = self.critic_1.named_parameters()
        critic_2_params = self.critic_2.named_parameters()
        target_actor_params = self.target_actor.named_parameters()
        target_critic_1_params = self.target_critic_1.named_parameters()
        target_critic_2_params = self.target_critic_2.named_parameters()

        critic_1 = dict(critic_1_params)
        critic_2 = dict(critic_2_params)
        actor = dict(actor_params)
        target_actor = dict(target_actor_params)
        target_critic_1 = dict(target_critic_1_params)
        target_critic_2 = dict(target_critic_2_params)

        for name in critic_1:
            critic_1[name] = tau * critic_1[name].clone() + (1-tau) *\
                             target_critic_1[name].clone()

        for name in critic_2:
            critic_2[name] = tau * critic_2[name].clone() + (1-tau) *\
                             target_critic_2[name].clone()

        for name in actor:
            actor[name] = tau*actor[name].clone() + \
                          (1-tau)*target_actor[name].clone()

        self.target_critic_1.load_state_dict(critic_1)
        self.target_critic_2.load_state_dict(critic_2)
        self.target_actor.load_state_dict(actor)

    def save_models(self):
        """
        Description,
            Save all the models into respective checkpoints.
        """
        self.actor.save_checkpoint()
        self.target_actor.save_checkpoint()
        self.critic_1.save_checkpoint()
        self.critic_2.save_checkpoint()
        self.target_critic_1.save_checkpoint()
        self.target_critic_2.save_checkpoint()

    def load_models(self):
        """
        Description,
            Loads all the model from respective checkpoints.
        """
        self.actor.load_checkpoint()
        self.target_actor.load_checkpoint()
        self.critic_1.load_checkpoint()
        self.critic_2.load_checkpoint()
        self.target_critic_1.load_checkpoint()
        self.target_critic_2.load_checkpoint()
