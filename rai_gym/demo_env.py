import string
from time import sleep
import libry as ry
import os
import gym
from gym import spaces
import numpy as np
import numba


# Add Robot relative frame
robot_pos_offset = np.array([0.0, 0.0, 0.59])

# Robot Joint Limits
joint_low = np.array([-2.8, -1.7, -2.8, -3.0, -2.8, -0.0, -2.])
joint_high = np.array([2.8, 1.7, 2.8, -0.0, 2.8, 3.7, 2.8])


@numba.jit(nopython=True)
def _update_q(q, J, Y):
    q += np.linalg.pinv(J) @ Y
    return q


class Environment(gym.Env):
    """Custom Environment that follows gym interface"""

    def __init__(self, reach_skill):
        super(Environment, self).__init__()

        # Load the robot configuration file
        self.K = ry.Config()

        # Add Objects at random position
        self.obj1 = self.K.addFrame("Obj1")
        self.obj1.setShape(ry.ST.cylinder, [.05, .02])
        self.obj1.setColor([1, 0, 0])
        self.obj1.setQuaternion(np.array([np.pi, 0, 0, 0]))
        self.obj1.setMass(1.0)
        self.obj1.setContact(-2)

        self.obj2 = self.K.addFrame("Obj2")
        self.obj2.setShape(ry.ST.cylinder, [.05, .02])
        self.obj2.setColor([0, 0, 1])
        self.obj2.setQuaternion(np.array([np.pi, 0, 0, 0]))
        self.obj2.setMass(1.0)
        self.obj2.setContact(-2)

        self.K.getFrameState()
        self.K.addFile(os.path.abspath('robot_scene/robot_scene.g'))

        self.S = ry.Simulation(self.K, ry.SimulatorEngine.bullet, True)
        self.frame = 'gripper'

        # Init. gym environment
        self.tau = 1e-2
        self.low = np.array([-0.12, -0.12, 0.2])
        self.high = np.array([0.12, 0.12, 0.8])

        # Register Agents
        self.reach_skill = reach_skill

        # Init. Goal Indicator
        self.goal_marker = self.K.addFrame("goal_marker")
        self.goal_marker.setShape(ry.ST.sphere, [0.02])
        self.goal_marker.setColor([0, 1, 0])

        # Init. reset
        self.reset()

    def _dead_sim(self, iterations=1000) -> None:
        for _ in range(iterations):
            self.S.step([], 0.01, ry.ControlMode.none)

    def _randomize_box_position(self) -> None:
        # Get Object
        f = self.K.getFrame("Obj1")

        # Reset Position & color of the box
        position = np.random.uniform(-1.5, 1.5, (3,))
        position = np.clip(position,
                           np.array([-0.14, 0, 0.9]),
                           np.array([0.14, 0, 1.5]))
        f.setPosition(position)
        f.setQuaternion([np.pi, 0, 0, 0])

        # Get Object
        f = self.K.getFrame("Obj2")

        # Reset Position & color of the box
        position = np.random.uniform(-1.5, 1.5, (3,))
        position = np.clip(position,
                           np.array([0, 0.14, 0.9]),
                           np.array([0, 0.14, 1.5]))
        f.setPosition(position)
        sleep(0.01)
        f.setQuaternion([np.pi, 0, 0, 0])

        self.S.setState(self.K.getFrameState())
        self.S.step([], 0.01, ry.ControlMode.none)

    def _robot_reset(self, factor: float = 0.1) -> None:
        # Generate Random Robot pose and set
        q = np.random.rand(joint_low.shape[0],)
        smooth_q = factor * np.clip(q, joint_low, joint_high)
        self.K.setJointState(smooth_q)

    def _robot_step(self, action: np.ndarray) -> None:
        self.K.selectJoints(["finger1", "finger2"], True)

        action += robot_pos_offset
        F = self.K.feature(ry.FS.position, [self.frame])
        y1, J1 = F.eval(self.K)

        while not np.allclose(action, y1):
            q = np.array(self.K.getJointState())
            F = self.K.feature(ry.FS.position, [self.frame])
            y1, J1 = F.eval(self.K)

            F = self.K.feature(ry.FS.scalarProductZZ, [
                self.frame, "panda_link0"])
            y2, J2 = F.eval(self.K)

            Y = np.hstack((action - y1, 1.0 - y2))
            J = np.vstack((J1, J2))

            q = _update_q(q, J, Y)
            self.S.step(q, self.tau, ry.ControlMode.position)
            sleep(0.05)

    def MoveP2P(self, goal: np.ndarray, correct_pos=False) -> np.float:
        # Read Gripper Position
        F = self.K.feature(ry.FS.position, [self.frame])
        y = F.eval(self.K)[0]

        # Add 'Goal Marker' to the 'target'
        self.goal_marker.setPosition(goal)

        # Solve trajectory using agent to reach the target
        for _ in range(5):
            F = self.K.feature(ry.FS.position, [self.frame])
            y = F.eval(self.K)[0]
            state = np.concatenate((y, goal))
            action = self.reach_skill.choose_action(state)
            self._robot_step(action)

        error = np.linalg.norm(goal - y) * 1e3

        # Correct goal position
        if correct_pos:
            for _ in range(3):
                self._robot_step(goal - robot_pos_offset)

        return error

    def Grasp(self, state: string) -> None:
        q = self.S.get_q()

        # Gripping Close
        if state == 'close':
            self.S.closeGripper("gripper")

        # Gripping Close
        elif state == 'open':
            self.S.openGripper("gripper")

        else:
            raise ValueError(
                f'This grasp argument: {state} is not implemented')

        for _ in range(1000):
            self.S.step(q, self.tau, ry.ControlMode.position)

    def Wait(self, seconds: float):
        sleep(seconds * 1e-3)

    def reset(self) -> None:
        # Reset the state of the environment to an initial state
        self._randomize_box_position()
        self._dead_sim()
        self._robot_reset()
