import libry as ry
import os
import gym
from gym import spaces
from typing import Dict
import numpy as np
import numba


# Robot Joint Limits
joint_low = np.array([-2.8, -1.7, -2.8, -3.0, -2.8, -0.0])
joint_high = np.array([2.8, 1.7, 2.8, -0.0, 2.8, 3.7])


@numba.jit(nopython=True)
def _update_q(q, J, Y):
    q += np.linalg.pinv(J) @ Y
    return q


@numba.jit(nopython=True)
def _negative_distance(x, y):
    return -np.linalg.norm(y - x)


class Reach_Environment(gym.Env):
    """Custom Environment that follows gym interface"""

    def __init__(self, reward_type='sparse'):
        super(Reach_Environment, self).__init__()

        # Load the robot configuration file
        self.K = ry.Config()
        self.K.clear()
        self.K.addFile(os.path.abspath('robot_scene/robot_scene.g'))
        self.K.selectJoints(["finger1", "finger2"], True)
        self.S = ry.Simulation(self.K, ry.SimulatorEngine.bullet, True)
        self.frame = 'gripper'
        self.IK_steps = 5

        # Init. gym environment
        self.reward_type = reward_type
        self.max_episode_length = 250
        self.tau = 1e-2
        self.low = np.array([-0.15, -0.15, 0.2])
        self.high = np.array([0.15, 0.15, 0.7])

        # Init. spaces
        self.action_space = spaces.Box(
            low=self.low,
            high=self.high,
            dtype=np.float64)

        self.observation_space = spaces.Box(
            low=np.hstack((self.low, self.low)),
            high=np.hstack((self.high, self.high)),
            dtype=np.float64)

        # Init. focal length
        f = 0.895
        f = f * 360.0
        self.intrinsic = [f, f, 320.0, 180.0]

        # Init. Camera into Simulation World
        self.K.setJointState(np.zeros_like(joint_low))
        self.S.addSensor("camera")
        self.background_rgb, self.background_depth = self.S.getImageAndDepth()

        # Goal Indicator
        self.obj = self.K.addFrame("object")
        self.obj.setPosition([0, 0, 0.9])
        self.obj.setShape(ry.ST.ssBox, [.05, .05, .1, 0])
        self.obj.setColor([1, 0, 1])

        # Pre-Reset the env.
        self.random_target = np.zeros(3)
        self.reset()

        # S.addImp(ry.ImpType.objectImpulses, ['obj0'], [])

    def _random_config_space(self, factor=0.1):
        q = np.random.rand(joint_low.shape[0],)
        return factor * np.clip(q, joint_low, joint_high)

    def _random_pos_target(self) -> np.ndarray:
        """
        Returns a random task space reachable target

        Returns:
            np.ndarray: target
        """
        # Set Target in the task space
        self.K.setJointState(self._random_config_space())

        # Move the robot to random position
        position = np.random.uniform(-1.5, 1.5, (3,))
        position = np.clip(position, self.low, self.high)
        position[-1] += 0.59
        self.obj.setPosition(position)

        return self.obj.getPosition()

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
        signal = signal + np.array([0.0, 0.0, 0.59])
        F = self.K.feature(ry.FS.position, [self.frame])
        y1, J1 = F.eval(self.K)

        while not np.allclose(signal, y1):
            q = np.array(self.K.getJointState())
            F = self.K.feature(ry.FS.position, [self.frame])
            y1, J1 = F.eval(self.K)

            F = self.K.feature(ry.FS.scalarProductYZ, [
                               self.frame, "panda_link0"])
            y2, J2 = F.eval(self.K)

            Y = np.hstack((signal - y1, 1.0 - y2))
            J = np.vstack((J1, J2))

            q = _update_q(q, J, Y)

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
        done = False
        distance = _negative_distance(next_state, target_state)

        if distance >= 0.001:
            done = True
            reward = 0.0
        else:
            if self.reward_type == 'dense':
                reward = distance
            elif self.reward_type == 'sparse':
                reward = -1.0

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
        return self._current_state()
