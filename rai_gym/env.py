import libry as ry
import os
import gym
from gym import spaces
from typing import Dict
import numpy as np


class RAI_Env(gym.Env):
    """Custom Environment that follows gym interface"""

    def __init__(self, reward_type='sparse'):
        super(RAI_Env, self).__init__()

        # Load the robot configuration file
        self.K = ry.Config()
        self.K.clear()
        self.K.addFile(os.path.abspath('robot_scene/robot_scene.g'))
        self.S = ry.Simulation(self.K, ry.SimulatorEngine.bullet, True)
        self.frame = 'gripperCenter'
        self.IK_steps = 5

        # Init. gym environment
        self.reward_type = reward_type
        self.max_episode_length = 250
        self.tau = 1e-2

        # Init. spaces
        self.action_space = spaces.Box(
            low=np.array([-0.5, -0.5, 0.3]),
            high=np.array([0.5, 0.5, 0.5]),
            dtype=np.float64)

        self.observation_space = spaces.Box(
            low=np.array([-0.5, -0.5, 0.3, -0.5, -0.5, 0.3]),
            high=np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5]),
            dtype=np.float64)

        # Goal Indicator
        self.ball = self.K.addFrame(name="ball")
        self.ball.setShape(ry.ST.sphere, [.05])
        self.ball.setColor([0, 1, 0])

        # Pre-Reset the env.
        self.random_target = np.zeros(3)
        self.reset()

    def _random_config_space(self):
        q0 = np.random.uniform(-2.8, 2.8)
        q1 = np.random.uniform(-1.7, 1.7)
        q2 = np.random.uniform(-2.8, 2.8)
        q3 = np.random.uniform(-3.0, -0.0)
        q4 = np.random.uniform(-2.8, 2.8)
        q5 = np.random.uniform(-0.0, 3.7)
        q6 = np.random.uniform(-2.8, 2.8)

        return np.array([q0, q1, q2, q3, q4, q5, q6])

    def _random_pos_target(self) -> np.ndarray:
        """
        Returns a random task space reachable target

        Returns:
            np.ndarray: target
        """
        # Set Target in the task space
        self.K.setJointState(self._random_config_space())
        state = self._current_state()

        # Move the robot to random position
        self.K.setJointState(self._random_config_space())
        return np.clip(state['y'], np.array([-0.5, -0.5, 0.3]), np.array([0.5, 0.5, 0.5]))

    def _current_state(self) -> Dict:
        """
        Reads and Returns feature states of a robot frame.

        Args:
            frame(str): Frame of the robot. Defaults to 'gripperCenter'

        Returns:
            Diction
        """
        F = self.K.feature(ry.FS.position, [self.frame])
        y = F.eval(self.K)[0]

        state = {
            'y': y,
            'target': self.random_target
        }

        return state

    def _actuate(self, signal: np.ndarray):
        """
        Steps the environment using an actuating signal.

        Args:
            signal (np.ndarray): Actuating signal.
        """
        for _ in range(self.IK_steps):
            q = self.K.getJointState()
            F = self.K.feature(ry.FS.position, ['gripperCenter'])
            y, J = F.eval(self.K)

            q += np.linalg.pinv(J) @ (signal - y)

            self.S.step(q, self.tau, ry.ControlMode.position)

        self.current_episode += 1

    def compute_reward(self, next_state: np.ndarray, target_state: np.ndarray):
        """
        Compute Reward for the environment.

        Args:
            next_state (np.ndarray): Next State
            target_state (np.ndarray): Target State

        Returns:
            np.float64: Reward
            np.bool_: Done
        """

        if np.allclose(target_state, next_state):
            done = True
            reward = 0.0
        else:
            if self.reward_type == 'dense':
                reward = -np.linalg.norm(target_state - next_state)
                done = False
            elif self.reward_type == 'sparse':
                reward = -1.0
                done = False

        if self.current_episode == self.max_episode_length:
            done = True

        return reward, done

    def step(self, action: np.ndarray):
        """
        Steps the env. by a step based on an action.

        Args:
            action (np.ndarray): Actuation signal.
        """
        self._actuate(action)
        next_state = self._current_state()

        reward, done = self.compute_reward(
            next_state['y'], self.random_target)

        return next_state, reward, done

    def reset(self):
        # Reset the state of the environment to an initial state
        self.current_episode = 0
        self.done = False
        self.random_target = self._random_pos_target()
        self.ball.setPosition(self.random_target)
        return self._current_state()
