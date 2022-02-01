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
    def __init__(self, extrensic: np.ndarray, filters: List[np.array]):
        """Initiates the class with intrinsic & extrensic camera params.

        Args:
            extrensic (np.ndarray): Camera Extrensic Matrix
            filters (List[np.array]): List of RGB Values to filter.
        """
        self.extrensic = extrensic
        self.filters = filters

    @staticmethod
    def _distance(input_vector: np.array, output_vector: np.array):
        return np.linalg.norm(output_vector - input_vector)

    def __call__(self, RGB: np.ndarray, Depth: np.ndarray):
        pass
