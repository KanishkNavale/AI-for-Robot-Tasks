import libry as ry
from time import sleep
import gym
from gym import spaces
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
        self.max_episode_length = 250
        self.tau = 0.0001
        self.action_space = spaces.Box(
            low=np.array([-2.8973, -1.7628, -2.8973, -
                         3.0718, -2.8973, -0.0175, -2.8973]),
            high=np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973]), dtype=np.float32)

        self.observation_space = spaces.Box(
            low=np.array(
                [-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973,
                 -1.0, -1.0, -1.0, -1.0, -1.0, -1.0]),
            high=np.array(
                [2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973,
                 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]), dtype=np.float32)

        # Goal Indicator
        self.ball = self.K.addFrame(name="ball")
        self.ball.setShape(ry.ST.sphere, [.05])
        self.ball.setColor([0, 1, 0])

        # Pre-Reset the env.
        self.random_target = np.zeros(3).astype(np.float32)
        self.reset()

    def _expert_action(self):
        state = self._current_state()
        y = state['y']
        J = state['J']
        q = state['q']

        q += np.linalg.pinv(J) @ (self.random_target - y)
        return q

    def _random_config_space(self):
        q0 = np.random.uniform(-2.8973, 2.8973)
        q1 = np.random.uniform(-1.7628, 1.7628)
        q2 = np.random.uniform(-2.8973, 2.8973)
        q3 = np.random.uniform(-3.0718, -0.0698)
        q4 = np.random.uniform(-2.8973, 2.8973)
        q5 = np.random.uniform(-0.0175, 3.7525)
        q6 = np.random.uniform(-2.8973, 2.8973)

        return np.array([q0, q1, q2, q3, q4, q5, q6]).astype(np.float32)

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
        return state['y']

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
        y, J = F.eval(self.K)

        state = {
            'q': q,
            'y': y,
            'J': J,
            'target': self.random_target
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

    def _compute_reward(self, action, next_state: np.ndarray,
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
        exp_act = self._expert_action()
        reward = -np.square(np.subtract(exp_act, action)).mean()

        if np.allclose(self.random_target, next_state):
            done = True
        elif self.current_episode == self.max_episode_length:
            done = True
        else:
            done = False

        return reward.astype(np.float32), done

    def step(self, action: np.ndarray):
        """
        Steps the env. by a step based on an action.

        Args:
            action (np.ndarray): Actuation signal.
        """
        self._actuate(action)
        next_state = self._current_state()

        reward, done = self._compute_reward(
            action, next_state['y'], self.random_target)

        state = np.hstack(
            (next_state['q'], next_state['y'], next_state['target'])).astype(np.float32)
        return state.astype(np.float64), reward, done, {}

    def reset(self):
        # Reset the state of the environment to an initial state
        self.current_episode = 0
        self.done = False
        self.random_target = self._random_pos_target()
        self.ball.setPosition(self.random_target)
        state = self._current_state()
        return np.hstack((state['q'], state['y'], state['target'])).astype(np.float32)
