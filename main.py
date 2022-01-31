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

    # Get Strategy
    object = env.Obj1.getPosition()
    bin = np.array([0, 0, object[-1]])

    # Tend the Object
    env.TendObject(object, bin)

    # Get Strategy
    object = env.Obj2.getPosition()
    bin = np.array([0, 0, object[-1]])

    # Tend the Object
    env.TendObject(object, bin)
