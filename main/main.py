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

    # Execute the app
    for _ in range(10):
        env.run()
