from typing import List, Union
import string

import os
import gym
import numpy as np
from time import sleep
import numba
import json
import cv2

from vision_pipeline.pose import PosePredictor, Object

import libry as ry


# Add Robot relative frame
robot_pos_offset = np.array([0.0, 0.0, 0.59])


@numba.jit(nopython=True)
def _update_q(q, J, Y):
    """Computes the 'q' update faster.

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

        # Add Camera
        self.S.addSensor("camera")
        self.camera = self.K.frame("camera")
        try:
            self.S.getImageAndDepth()
        except Exception as e:
            raise ValueError(f'Camera Fault:{e}')

        # Setup Camera Params.
        f = 0.895 * 360.0
        self.intrinsic = [f, f, 320.0, 180.0]

        cam_rotation = self.camera.getRotationMatrix()
        cam_translation = self.camera.getPosition()

        cam_rotation_translation = np.hstack(
            (cam_rotation, cam_translation[:, None]))
        cam_rigid_transformation = np.vstack(
            (cam_rotation_translation, [0, 0, 0, 1]))

        try:
            assert cam_rigid_transformation.shape == (4, 4)
        except:
            raise ValueError(
                "Error building the Camera's Rigid Body Transformation Matrix")

        # Reset Robot
        self._robot_reset()
        self._dead_sim()

        # Init. PosePredictor App.
        self.pose_predictor = PosePredictor(cam_rigid_transformation)

        # Init. Goal Indicator
        self.goal_marker = self.K.addFrame("goal_marker")
        self.goal_marker.setShape(ry.ST.sphere, [0.02])
        self.goal_marker.setColor([0, 1, 0])

        # Add object in the env.
        self.Obj1 = self.K.getFrame("Obj1")
        self.Obj2 = self.K.getFrame("Obj2")

        # Assign terminal steps for IK
        self.IK_Steps = 5

        # Init. reset
        self._reset()

    def _dead_sim(self, iterations: int = 1000) -> None:
        """Runs a no control simulation to refresh the states of the scene.

        Args:
            iterations (int, optional): simulation steps. Defaults to 1000.
        """
        for _ in range(iterations):
            self.S.step([], 0.01, ry.ControlMode.none)

    def _randomize_box_position(self) -> None:
        # Reset Position & color of the box
        position = np.random.uniform(-1.5, 1.5, (3,))
        position = np.clip(position,
                           np.array([-0.14, -0.14, 1.1]),
                           np.array([0, 0, 1.1]))
        f = self.K.getFrame("Obj1")
        f.setPosition(position)
        self.S.setState(self.K.getFrameState())
        self.S.step([], 0.01, ry.ControlMode.none)

        position = np.random.uniform(-1.5, 1.5, (3,))
        position = np.clip(position,
                           np.array([0, 0, 1.1]),
                           np.array([0.14, 0.14, 1.1]))
        f = self.K.getFrame("Obj2")
        f.setPosition(position)
        self.S.setState(self.K.getFrameState())
        self.S.step([], 0.01, ry.ControlMode.none)

    def _robot_reset(self, factor: float = 0.1) -> None:
        """Resets the robot to randomized poisition close to zero.

        Args:
            factor (float, optional): smoothing factor. Defaults to 0.1.
        """
        # Generate Random Robot pose and set
        q = np.ones(len(self.K.getJointState()))
        smooth_q = factor * q
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

    def _MoveP2P(self, goal: np.ndarray, correct=False) -> np.float:
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

    def _Grasp(self, state: string) -> None:
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

    def _Wait(self, seconds: float):
        """Makes the simulation wait for defined seconds

        Args:
            seconds (float): seconds
        """
        sleep(seconds * 1e-3)

    def _TendObject(self, pick: float, drop: float) -> None:
        """Performs a pick and drop cycle as per industrial standards.

        Args:
            pick (float): Object Pick-Up Location
            drop (float): Object Drop-Off Location
        """
        self._Grasp('open')

        self._MoveP2P(pick + np.array([0.0, 0.0, .10]))
        self._MoveP2P(pick, True)
        self._Grasp('close')
        self._MoveP2P(pick + np.array([0.0, 0.0, .10]))

        self._MoveP2P(drop + np.array([0.0, 0.0, .10]))
        self._MoveP2P(drop, True)
        self._Grasp('open')
        self._MoveP2P(drop + np.array([0.0, 0.0, .10]))

    def _ComputeStrategy(self) -> Union[List[Object], np.ndarray]:
        """Computes the Object Tending Strategy using Camera.

        Returns:
            Union[List[Object], np.ndarray]: List of objects with their tending strategy, processed image
        """

        rgb_image, depth_data = self.S.getImageAndDepth()
        point_cloud = self.S.depthData2pointCloud(depth_data, self.intrinsic)
        self.camera.setPointCloud(rgb_image, point_cloud)

        return self.pose_predictor(rgb_image, point_cloud)

    def _ExecuteStrategy(self, objects: List[Object]):
        """Executes computed strategy

        Args:
            List ([type]): List of objects with their tending strategy
        """
        box1 = self.K.getFrame("box1").getPosition() + np.array([0, 0, 0.07])
        box2 = self.K.getFrame("box2").getPosition() + np.array([0, 0, 0.07])

        for object in objects:
            if object.color == 'red':
                self._TendObject(object.world_coord, box2)
                self._Wait(2)
            elif object.color == 'blue':
                self._TendObject(object.world_coord, box1)
                self._Wait(2)
            else:
                raise ValueError('ERR: Found unregistered color coded object')

    def _dump_debug(self, folder_location: string, objects: List[Object], image: np.ndarray):
        """Dumps app processed data into debug folder

        Args:
            objects (List[Object]): List of 'Object' dataclass
            image (np.ndarray); Processed RGB Image H x W x 3
        """
        folder = os.path.abspath(folder_location)

        # Create data
        data = []
        for object in objects:
            object_data = {'Object ID': object.id,
                           'Camera Coordinates [u, v]': object.image_coord,
                           'World Coordinates [x, y, z]': object.world_coord.tolist(),
                           'Color': object.color
                           }
            data.append(object_data)

        # Dump .json file
        file = os.path.join(folder, 'object_data.json')
        with open(file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

        # Dump the processed image
        font = cv2.FONT_HERSHEY_SIMPLEX
        green = (0, 255, 0)

        id = objects[0].id
        uvs = objects[0].image_coord
        pose = np.around(objects[0].world_coord, 2)
        color = objects[0].color

        cv2.putText(image, f'Object ID: {id}',
                    (10, 20), font, 0.35, green, 1, cv2.LINE_AA)
        cv2.putText(image, f'Image uvs: {uvs}',
                    (10, 40), font, 0.35, green, 1, cv2.LINE_AA)
        cv2.putText(image, f'World pose: {pose}',
                    (10, 60), font, 0.35, green, 1, cv2.LINE_AA)
        cv2.putText(image, f'Color: {color}',
                    (10, 80), font, 0.35, green, 1, cv2.LINE_AA)

        id = objects[1].id
        uvs = objects[1].image_coord
        pose = np.around(objects[1].world_coord, 2)
        color = objects[1].color

        cv2.putText(image, f'Object ID: {id}',
                    (10, 275), font, 0.35, green, 1, cv2.LINE_AA)
        cv2.putText(image, f'Image uvs: {uvs}',
                    (10, 295), font, 0.35, green, 1, cv2.LINE_AA)
        cv2.putText(image, f'World pose: {pose}',
                    (10, 315), font, 0.35, green, 1, cv2.LINE_AA)
        cv2.putText(image, f'Color: {color}',
                    (10, 335), font, 0.35, green, 1, cv2.LINE_AA)

        file = os.path.join(folder, 'processed_image.png')
        cv2.imwrite(file, image)

    def _reset(self) -> None:
        # Resets the environment.
        self._randomize_box_position()
        self._dead_sim()
        self._robot_reset()

    def run(self) -> None:
        # Executes the tending process
        self._reset()
        object_list, processed_rgb = self._ComputeStrategy()
        self._ExecuteStrategy(object_list)
        self._dump_debug('main/data', object_list, processed_rgb)
