import libry as ry
from time import sleep
import gym
from typing import Dict
import numpy as np


class RAI_Env(gym.Env):
    """Custom Environment that follows gym interface"""

    def __init__(self):
        super(RAI_Env, self).__init__()

        # Load the robot configuration file
        self.K = ry.Config()
        self.K.clear()
        self.K.addFile('panda/panda_fixGripper.g')
        self.S = ry.Simulation(self.K, ry.SimulatorEngine.bullet, True)

        # Init. gym environment
        self.max_episode_length = 1000
        self.tau = 0.01

        # Goal Indicator
        self.ball = self.K.addFrame(name="ball")
        self.ball.setShape(ry.ST.sphere, [.05])
        self.ball.setColor([0, 1, 0])

        # Pre-Reset the env.
        self.reset()

    def _current_state(self, frame: str = 'gripperCenter') -> Dict:
        """
        Reads and Returns feature states of a robot frame.

        Args:
            frame(str): Frame of the robot. Defaults to 'gripperCenter'

        Returns:
            Diction
        """
        q = self.K.getJointState()
        F = self.K.feature(ry.FS.position, [frame])
        y = F.eval(self.K)[0]

        state = {
            'q': q,
            'y': y
        }

        return state

    def _random_action(self) -> np.ndarray:
        """
        Returns a random construct of action signal

        Returns:
            np.ndarray: Random action signal.
        """
        return(np.random.uniform(-np.pi, np.pi, (7,)))

    def _actuate(self, signal: np.ndarray):
        """
        Steps the environment using an actuating signal.

        Args:
            signal (np.ndarray): Actuating signal.
        """
        self.S.step(signal, self.tau, ry.ControlMode.position)
        self.current_episode += 1
        sleep(self.tau)

    def _compute_reward(self, next_state: np.ndarray,
                        target_state: np.ndarray):
        """
        Compute Reward for the environment.

        Args:
            next_state (np.ndarray): Next State
            target_state (np.ndarray): Target State

        Returns:
            np.float64: Reward
            np.bool_: Done
        """
        reward = -np.linalg.norm(target_state - next_state)

        if reward >= -0.5:
            done = True
        elif self.current_episode == self.max_episode_length:
            done = True
        else:
            done = False

        return reward, done

    def step(self, action: np.ndarray):
        """
        Steps the env. by a step based on an action.

        Args:
            action (np.ndarray): Actuation signal.
        """
        self._actuate(action)
        next_state = self._current_state()

        reward, done = self._compute_reward(
            next_state['y'], self.random_target)

        return reward, done

    def reset(self):
        # Reset the state of the environment to an initial state
        self.current_episode = 0
        self.done = False
        self.random_target = np.random.uniform(np.pi / 2, 0, (3,))
        self.ball.setPosition(self.random_target)
