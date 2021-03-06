# Library Imports
import gym
import numpy as np
import os
import copy
from DDPG import Agent

# Main script pointer
if __name__ == "__main__":

    # Init. datapath
    data_path = os.path.dirname(os.path.abspath(__file__)) + '/data/'

    # Load the environment
    env = gym.make('FetchReach-v1')
    env.env.reward_type = "dense"

    # Build Shapes for States and Actions
    state_shape = env.observation_space['observation'].shape[0] + \
        env.observation_space['achieved_goal'].shape[0] + \
        env.observation_space['desired_goal'].shape[0]
    action_shape = env.action_space.shape[0]

    # Init. Training
    best_score = -np.inf
    score_history = []
    avg_history = []
    n_games = 3000

    # Init. Agent
    agent = Agent(env, state_shape, action_shape, data_path, n_games)

    for i in range(n_games):
        score = 0
        done = False

        # Initial Reset of Environment
        OBS = env.reset()

        while not done:
            # Unpack the observation
            state, curr_actgoal, curr_desgoal = OBS.values()
            obs = np.concatenate((state, curr_actgoal, curr_desgoal))

            # Choose agent based action & make a transition
            action = agent.choose_action(obs)
            next_OBS, reward, done, info = env.step(action)

            next_state, next_actgoal, next_desgoal = next_OBS.values()
            next_obs = np.concatenate((next_state, next_actgoal, next_desgoal))

            agent.store(obs, action, reward, next_obs, done)

            OBS = copy.deepcopy(next_OBS)
            score += reward

        # Optimize the agent
        agent.optimize()

        score_history.append(score)
        avg_score = np.mean(score_history[-100:])
        avg_history.append(avg_score)

        if avg_score > best_score:
            best_score = avg_score
            agent.save_models()
            print(f'Episode:{i}'
                  f'\t ACC. Rewards: {score:3.2f}'
                  f'\t AVG. Rewards: {avg_score:3.2f}'
                  f'\t *** MODEL SAVED! ***')
        else:
            print(f'Episode:{i}'
                  f'\t ACC. Rewards: {score:3.2f}'
                  f'\t AVG. Rewards: {avg_score:3.2f}')

        # Save the score log
        np.save(data_path + 'score_history', score_history, allow_pickle=False)
        np.save(data_path + 'avg_history', avg_history, allow_pickle=False)
