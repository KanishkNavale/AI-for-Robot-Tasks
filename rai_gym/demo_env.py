import string
from time import sleep
import libry as ry
import os
import gym
from gym import spaces
import numpy as np
import numba


# Robot Joint Limits
joint_low = np.array([-2.8, -1.7, -2.8, -3.0, -2.8, -0.0, -2.])
joint_high = np.array([2.8, 1.7, 2.8, -0.0, 2.8, 3.7, 2.8])

# Add Robot relative frame
robot_pos_offset = np.array([0.0, 0.0, 0.59])


@numba.jit(nopython=True)
def _update_q(q, J, Y):
    q += np.linalg.pinv(J) @ Y
    return q


class Environment(gym.Env):
    """Custom Environment that follows gym interface"""

    def __init__(self):
        super(Environment, self).__init__()

        # Load the robot configuration file
        self.K = ry.Config()

        # Add box for robot tending
        self.box = self.K.addFrame("box")
        self.box.setShape(ry.ST.ssBox, [.05, .05, .05, 0])
        self.box.setPosition([0, 0, 1.2])
        self.box.setMass(5.0)
        self.box.setContact(-1)

        self.K.getFrameState()
        self.K.addFile(os.path.abspath('robot_scene/robot_scene.g'))
        self.K.selectJoints(["finger1", "finger2"], True)

        self.S = ry.Simulation(self.K, ry.SimulatorEngine.bullet, True)
        self.frame = 'gripperCenter'

        # Init. gym environment
        self.tau = 1e-2
        self.low = np.array([-0.12, -0.12, 0.2])
        self.high = np.array([0.12, 0.12, 0.8])

        # Init. Shapes for ENV.
        self.action_space = spaces.Box(
            low=self.low,
            high=self.high,
            dtype=np.float64)

        self.observation_space = spaces.Box(
            low=np.hstack((self.low, self.low)),
            high=np.hstack((self.high, self.high)),
            dtype=np.float64)

        # Init. Goal Indicator
        self.goal_marker = self.K.addFrame("goal_marker")
        self.goal_marker.setShape(ry.ST.sphere, [0.02])
        self.goal_marker.setColor([0, 1, 0])

        # Pre-Reset the env.
        self.reset()

    def _dead_sim(self, iterations=1000) -> None:
        for _ in range(iterations):
            self.S.step([], 0.01, ry.ControlMode.none)

    def _randomize_box_color(self) -> None:
        # Color Scheme
        red = [1, 0, 0]
        blue = [0, 0, 1]
        colors = [red, blue]
        self.box.setColor(colors[np.random.randint(len(colors))])

    def _randomize_box_position(self) -> None:
        # Reset Position & color of the box
        position = np.random.uniform(-1.5, 1.5, (3,))
        position = np.clip(position,
                           np.array([-0.1, -0.1, 1.1]),
                           np.array([0.1, 0.05, 1.2]))
        f = self.K.getFrame("box")
        f.setPosition(position)
        self.S.setState(self.K.getFrameState())
        self.S.step([], 0.01, ry.ControlMode.none)

    def _robot_reset(self, factor: float = 0.1) -> None:
        # Generate Random Robot pose and set
        q = np.random.rand(joint_low.shape[0],)
        smooth_q = factor * np.clip(q, joint_low, joint_high)
        self.K.setJointState(smooth_q)

    def _robot_step(self, action: np.ndarray) -> None:
        action += robot_pos_offset
        F = self.K.feature(ry.FS.position, [self.frame])
        y1, J1 = F.eval(self.K)

        while not np.allclose(action, y1):
            q = np.array(self.K.getJointState())
            F = self.K.feature(ry.FS.position, [self.frame])
            y1, J1 = F.eval(self.K)

            F = self.K.feature(ry.FS.scalarProductYZ, [
                               self.frame, "panda_link0"])
            y2, J2 = F.eval(self.K)

            Y = np.hstack((action - y1, 1.0 - y2))
            J = np.vstack((J1, J2))

            q = _update_q(q, J, Y)

            self.S.step(q, self.tau, ry.ControlMode.position)
            sleep(self.tau * 10)

    def MoveP2P(self, reach_agent, goal: np.ndarray, correct_pos=False) -> np.float:
        # Read Gripper Position
        F = self.K.feature(ry.FS.position, [self.frame])
        y = F.eval(self.K)[0]

        # Add 'Goal Marker' to the 'target'
        self.goal_marker.setPosition(goal)

        # Solve trajectory using agent to reach the target
        for _ in range(10):
            F = self.K.feature(ry.FS.position, [self.frame])
            y = F.eval(self.K)[0]
            state = np.concatenate((y, goal))
            action = reach_agent.choose_action(state)
            self._robot_step(action)

        error = np.linalg.norm(goal - y) * 1e3

        # Correct goal position
        if correct_pos:
            for _ in range(3):
                self._robot_step(goal - robot_pos_offset)

        return error

    def Grasp(self, state: string) -> None:
        # Gripping Close
        if state == 'close':
            self.S.closeGripper("gripper")
            while not self.S.getGripperIsGrasping("gripper"):
                pass

        # Gripping Close
        elif state == 'open':
            self.S.openGripper("gripper")
            while self.S.getGripperIsGrasping("gripper"):
                pass

        else:
            raise ValueError(
                f'This grasp argument: {state} is not implemented')

    def reset(self) -> None:
        # Reset the state of the environment to an initial state
        self._randomize_box_color()
        self._randomize_box_position()
        self._dead_sim()
        self._robot_reset()
