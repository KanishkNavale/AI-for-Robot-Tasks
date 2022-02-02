from typing import List

import numpy as np
import cv2
from dataclasses import dataclass


@dataclass
class Object:
    id: int
    image_coord: np.array
    world_coord: np.array
    color: np.array


class PosePredictor:
    def __init__(self, extrensic: np.ndarray):
        """Initiates the class with intrinsic & extrensic camera params.

        Args:
            extrensic (np.ndarray): Camera Extrensic Matrix
            filters (List[np.array]): List of RGB Values to filter.
        """
        self.extrensic = extrensic
        self.filters = ['red', 'blue']

    def _color_filter(self, color, rgb):
        filter = np.zeros((rgb.shape[0], rgb.shape[1]), dtype=np.uint8)
        for u in range(rgb.shape[0]):
            for v in range(rgb.shape[1]):
                if color == 'red':
                    if rgb[u, v][2] < 25 and rgb[u, v][1] < 25:
                        filter[u, v] = 255
                elif color == 'blue':
                    if rgb[u, v][0] < 25 and rgb[u, v][1] < 25:
                        filter[u, v] = 255
        return filter

    def _compute_centers(self, filtered, rgb, pcl):
        cnts, _ = cv2.findContours(
            filtered, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for c in cnts:
            # compute the center of the contour
            M = cv2.moments(c)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])

            # draw the contour and center of the shape on the image
            cv2.drawContours(rgb, [c], -1, (0, 255, 0), 2)
            cv2.circle(rgb, (cX, cY), 2, (0, 255, 0), -1)

            # Find mean red depth
            mean = np.zeros(3)
            for pixel in c:
                pixel_X = pixel[0, 1]
                pixel_Y = pixel[0, 0]
                mean += pcl[pixel_X, pixel_Y]
            mean /= c.shape[0]

        return [cX, cY], mean

    def __call__(self, rgb: np.ndarray, pcl: np.ndarray):

        # List of Objects
        object_data = []

        # Find Red and Blue Object
        for i, color in enumerate(self.filters):
            filtered = self._color_filter(color, rgb)
            uvs, local_pose = self._compute_centers(filtered, rgb, pcl)
            pose = self.extrensic @ np.hstack((local_pose, 1))
            object_data.append(Object(id=i, image_coord=uvs,
                               world_coord=pose[:3], color=color))

        return object_data, rgb
