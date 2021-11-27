# Library Imports
import gym
import numpy as np
import sys
import os
import copy

# Import TD3.Agent
sys.path.append('robotics-lab-project/rl_agents')
from TD3 import Agent

# Definition for Dense Reward Engineering
def dense_reward(desired_goal, achieved_goal):
    """
    Description:
        Generates a dense reward for the given states.
    Args:
        desired_goal ([type=np.float32, size=(desired_goal.shape,)]): Desired Goal Position
        achieved_goal ([type]): [description]
    Returns:
        [np.float32]: Dense Reward
    """
    scale = 10

    if np.array_equal(desired_goal, achieved_goal):
        reward = 1 * scale
        done = True
    else:
        reward = -np.linalg.norm(desired_goal - achieved_goal) * scale
        done = False

    return reward, done


# Main script pointer
if __name__ == "__main__":
    # Init. datapath
    data_path = os.getcwd() + '/robotics-lab-project/proof_of_concept/reach/data/'

    # Load the environment
    env = gym.make('FetchReach-v1')
    
    # Build Shapes for States and Actions
    state_shape = env.observation_space['observation'].shape[0] + \
                  env.observation_space['achieved_goal'].shape[0] + \
                  env.observation_space['desired_goal'].shape[0]
    action_shape = env.action_space.shape[0]

    # Init. Agent
    agent = Agent(state_shape, action_shape, data_path)

    # Init. Training
    best_score = -np.inf
    score_history = []
    avg_history = []
    n_games = 3000

    for i in range(n_games):
        score = 0
        done = False

        # Initial Reset of Environment
        OBS = env.reset()

        tick = 0
        while not done:
            # Unpack the observation
            state, curr_actgoal, curr_desgoal = OBS.values()
            obs = np.concatenate((state, curr_actgoal, curr_desgoal))

            # Choose agent based action & make a transition
            action = agent.choose_action(obs)
            next_OBS, reward, done, info = env.step(action)

            next_state, next_actgoal, next_desgoal = next_OBS.values()
            next_obs = np.concatenate((next_state, next_actgoal, next_desgoal))

            reward, done = dense_reward(next_desgoal, curr_actgoal)
            agent.memory.store_transition(np.concatenate((state, curr_actgoal, curr_desgoal)),
                                          action,
                                          reward,
                                          np.concatenate((next_state, next_actgoal, next_desgoal)),
                                          done)

            OBS = copy.deepcopy(next_OBS)
            score += reward
            tick += 1

            if tick == env._max_episode_steps:
                break

        # Optimize the agent
        agent.optimize()

        score_history.append(score)
        avg_score = np.mean(score_history[-100:])
        avg_history.append(avg_score)

        if avg_score > best_score:
            best_score = avg_score
            agent.save_models()
            print(f'Episode:{i} \t ACC. Rewards: {score:3.2f} \t AVG. Rewards: {avg_score:3.2f} \t *** MODEL SAVED! ***')
        else:
            print(f'Episode:{i} \t ACC. Rewards: {score:3.2f} \t AVG. Rewards: {avg_score:3.2f}')

        # Save the score log
        np.save(data_path + 'score_history', score_history, allow_pickle=False)
        np.save(data_path + 'avg_history', avg_history, allow_pickle=False)