import libry as ry
import os
import gym
from gym import spaces
from typing import Dict
import numpy as np
import numba


@numba.jit(nopython=True)
def _update_q(q, J, Y):
    q += np.linalg.pinv(J) @ Y
    return q


class Grasp_Environment(gym.Env):
    """Custom Environment that follows gym interface"""

    def __init__(self):
        super(Grasp_Environment, self).__init__()

        # Init. gym environment
        self.max_episode_length = 250
        self.tau = 1e-2
        self.low = np.array([-0.12, -0.12])
        self.high = np.array([0.12, 0.12])

        # Init. spaces
        self.action_space = spaces.Box(
            low=self.low,
            high=self.high,
            dtype=np.float64)

        self.observation_space = spaces.Box(
            low=np.hstack((self.low, self.low)),
            high=np.hstack((self.high, self.high)),
            dtype=np.float64)

        # Load Scene
        self._construct_scene()

    def _construct_scene(self):
        self.RealWorld = ry.Config()
        self.RealWorld.addFile(os.path.abspath('robot_scene/robot_scene.g'))

        self.K = ry.Config()
        self.K.addFile(os.path.abspath('robot_scene/robot_scene.g'))

        # Link Object to Simulation
        self.obj1 = self.RealWorld.addFrame("object")
        self.obj1.setMass(1)
        self.obj1.setPosition([0.0, 0.03, 1.2])
        self.obj1.setShape(ry.ST.ssBox, [.05, .05, .05, 0])
        self.obj1.setColor([1, 0, 0])
        self.obj1.setContact(True)

        # Init. Simulation to RealWorld
        self.S = self.RealWorld.simulation(ry.SimulatorEngine.bullet, True)

        self.obj2 = self.K.addFrame("object")
        self.obj2.setMass(1)
        self.obj2.setPosition([0.0, 0.03, 1.2])
        self.obj2.setShape(ry.ST.ssBox, [.05, .05, .05, 0])
        self.obj2.setColor([1, 0, 0])
        self.obj2.setContact(True)

        for _ in range(1000):
            self.S.step([], 0.01, ry.ControlMode.none)

        self.frame = 'gripperCenter'

    def _align_robot(self):
        self.K.selectJoints(["finger1", "finger2"], True)
        target = self.obj1.getPosition()

        F = self.K.feature(ry.FS.position, [self.frame])
        y1, J1 = F.eval(self.K)

        while not np.allclose(target, y1):
            q = np.array(self.K.getJointState())
            F = self.K.feature(ry.FS.position, [self.frame])
            y1, J1 = F.eval(self.K)
            print(target, y1)

            F = self.K.feature(ry.FS.scalarProductYZ, [
                               self.frame, "panda_link0"])
            y2, J2 = F.eval(self.K)

            Y = np.hstack((target - y1, 1.0 - y2))
            J = np.vstack((J1, J2))

            q += np.linalg.pinv(J) @ Y

            self.S.step(q, self.tau, ry.ControlMode.position)

        self.current_episode += 1

    def step(self, action: np.ndarray):
        """
        Steps the env. by a step based on an action.

        Args:
            action (np.ndarray): Actuation signal.
        """
        self._actuate(action)
        next_state = self._current_state()
        return next_state, 0, False

    def reset(self):
        # Reset the state of the environment to an initial state
        self.current_episode = 0
        self.done = False

        # Reset Position of the cube
        f = self.RealWorld.getFrame("object")
        position = self.obj1.getPosition()
        position[0] = np.random.uniform(-0.15, -0.05)
        position[1] = np.random.uniform(-0.15, -0.05)
        f.setPosition(position)
        self.S.setState(self.RealWorld.getFrameState())
        self.S.step([], self.tau, ry.ControlMode.none)

        # Reach for Grasping
        self._align_robot()
