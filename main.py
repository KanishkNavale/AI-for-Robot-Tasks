import numpy as np
import os

from rai_gym.demo_env import Environment
from rl_agents.DDPG import Agent


if __name__ == '__main__':
    # Init. DDPG Agent
    agent_path = os.path.abspath("trained_agents")
    reach_agent = Agent()
    reach_agent.actor.load_model(agent_path + '/')

    # Init. Environments
    env = Environment(reach_skill=reach_agent)
    env.reset()
    # env.run()

    red, blue = env._ComputeStrategy()
    pass

    # Get Strategy
    object = env.Obj1.getPosition()
    bin = np.array([0, 0, object[-1]])

    # Tend the Object
    env._TendObject(object, bin)

    # Get Strategy
    object = env.Obj2.getPosition()
    bin = np.array([0, 0, object[-1]])

    # Tend the Object
    env._TendObject(object, bin)
