import os
import string
import gym
import numpy as np
from time import sleep
import numba

import libry as ry


# Add Robot relative frame
robot_pos_offset = np.array([0.0, 0.0, 0.59])

# Robot Joint Limits
joint_low = np.array([-2.8, -1.7, -2.8, -3.0, -2.8, -0.0, -2.])
joint_high = np.array([2.8, 1.7, 2.8, -0.0, 2.8, 3.7, 2.8])


@numba.jit(nopython=True)
def _update_q(q, J, Y):
    """Numba Function!
       Computes the 'q' update faster.

    Args:
        q ([float]): Robot configuration space position.
        J ([float]): Robot Jacobian
        Y ([float]): Position Error

    Returns:
        [float]: Updated robot configuration space position.
    """
    q += np.linalg.pinv(J) @ Y
    return q


class Environment(gym.Env):
    """Custom Environment that follows gym interface"""

    def __init__(self, reach_skill):
        super(Environment, self).__init__()

        # Load the robot configuration file
        self.K = ry.Config()

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

        # Embed object in the scene
        self.Obj1 = self.K.getFrame("Obj1")
        self.Obj2 = self.K.getFrame("Obj2")

        # Assign terminal steps for IK
        self.IK_Steps = 5

        # Init. reset
        self.reset()

    def _dead_sim(self, iterations: int = 1000) -> None:
        """Runs a no control simulation to refresh the states of the scene.

        Args:
            iterations (int, optional): simulation steps. Defaults to 1000.
        """
        for _ in range(iterations):
            self.S.step([], 0.01, ry.ControlMode.none)

    def _robot_reset(self, factor: float = 0.1) -> None:
        """Resets the robot to randomized poisition close to zero.

        Args:
            factor (float, optional): smoothing factor. Defaults to 0.1.
        """
        # Generate Random Robot pose and set
        q = np.random.rand(joint_low.shape[0],)
        smooth_q = factor * np.clip(q, joint_low, joint_high)
        self.K.setJointState(smooth_q)

    def _robot_step(self, action: np.ndarray) -> None:
        """Moves the robot using IK to position specified by action

        Args:
            action (np.ndarray): Target Position P[x, y, z,]
        """
        self.K.selectJoints(["finger1", "finger2"], True)

        action += robot_pos_offset
        F = self.K.feature(ry.FS.position, [self.frame])
        y1, J1 = F.eval(self.K)

        for _ in range(self.IK_Steps):
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
            sleep(0.01)

    def MoveP2P(self, goal: np.float64, correct=False) -> np.float:
        """Use Deep Reinforcement Learning Agent to move robot to goal position.

        Args:
            goal (np.float64): Target Position P[x, y, z,]

        Returns:
            np.float: Position Error
        """
        # Read Gripper Position
        F = self.K.feature(ry.FS.position, [self.frame])
        y = F.eval(self.K)[0]

        # Add 'Goal Marker' to the 'target'
        self.goal_marker.setPosition(goal)

        # Solve trajectory using agent to reach the target
        for _ in range(self.IK_Steps):
            F = self.K.feature(ry.FS.position, [self.frame])
            y = F.eval(self.K)[0]
            state = np.concatenate((y, goal))
            action = self.reach_skill.choose_action(state)
            self._robot_step(action)

        error = np.linalg.norm(goal - y) * 1e3

        if correct:
            for _ in range(self.IK_Steps):
                self._robot_step(goal - robot_pos_offset)

        return error

    def Grasp(self, state: string) -> None:
        """Actuates the robot gripper to close and open.

        Args:
            state (string): State of the gripper.

        Raises:
            ValueError: Handle args. rather than open and close.
        """
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
            self.S.step([], 0.01, ry.ControlMode.none)

    def Wait(self, seconds: float):
        """Makes the simulation wait for defined seconds

        Args:
            seconds (float): seconds
        """
        sleep(seconds * 1e-3)

    def TendObject(self, pick: float, drop: float) -> None:
        """Performs a pick and drop cycle as per industrial standards.

        Args:
            pick (float): Object Pick-Up Location
            drop (float): Object Drop-Off Location
        """
        self.Grasp('open')

        self.MoveP2P(pick + np.array([0.0, 0.0, .10]))
        self.MoveP2P(pick, True)
        self.Grasp('close')
        self.MoveP2P(pick + np.array([0.0, 0.0, .10]))

        self.MoveP2P(drop + np.array([0.0, 0.0, .10]))
        self.MoveP2P(drop, True)
        self.Grasp('open')
        self.MoveP2P(drop + np.array([0.0, 0.0, .10]))

    def reset(self) -> None:
        """ Resets the environment.
        """
        self._dead_sim()
        self._robot_reset()
