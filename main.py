import numpy as np
import os

from rai_gym.demo_env import Environment
from rl_agents.DDPG import Agent

if __name__ == '__main__':
    # Init. agent for reaching tasks
    agent_path = os.path.abspath("trained_agents")

    low = np.array([-0.15, -0.15, 0.2])
    high = np.array([0.15, 0.15, 0.6])

    reach_agent = Agent(input_dims=6, n_actions=3, max_action=high,
                        min_action=low, datapath=None, n_games=1000)
    reach_agent.actor.load_model(agent_path + '/')

    # Init. Environments
    env = Environment(reach_skill=reach_agent)
    env.reset()

    # Stack of AI stacks
    env.Grasp(state='open')
    env.Wait(seconds=1)

    target = env.obj1.getPosition()
    env.MoveP2P(target + np.array([0.0, 0.0, .10]))
    env.Wait(seconds=1)

    env.MoveP2P(target)
    env.Wait(seconds=1)

    env.Grasp(state='close')
    env.Wait(seconds=1)
    env.MoveP2P(target + np.array([0.0, 0.0, .10]))

    bin_location = np.array([0, -0.12, 1.0])
    env.MoveP2P(bin_location + np.array([0.0, 0.0, .10]))
    env.Wait(seconds=1)

    env.MoveP2P(bin_location)
    env.Wait(seconds=1)

    env.Grasp(state='open')
    env.Wait(seconds=5)
