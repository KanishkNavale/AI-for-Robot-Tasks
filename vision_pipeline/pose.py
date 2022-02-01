from typing import List

import numpy as np
import cv2 as cv
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
        self.color = color

    def get_table_idx(Depth):
	idx_list = np.arange(Depth.shape[1])[Depth[225,:]==np.min(Depth[225,:])]
	idx_left, idx_right = idx_list[0],idx_list[-1]
	Depth_table_left = Depth[:,idx_left]
	idx_list_2 = np.arange(Depth.shape[0])[Depth_table_left==np.min(Depth_table_left)]
	Depth_left_diff = Depth_table_left[1:] - Depth_table_left[:-1]
	idx_top, idx_bottom = np.argmin(Depth_left_diff), np.argmax(Depth_left_diff)

	return idx_top, idx_bottom, idx_left, idx_right
	
    def get_coordinates(img, color, channel_helper):
	result = img.copy()
	lower = np.array([0])
	upper = np.array([25])
	channel = channel_helper[color]
	mask = cv.inRange(img[:,:,channel], lower, upper)
	idx_img = np.argwhere(mask==255)
	(y_min, x_min), (y_max, x_max) = np.min(idx_img, axis=0), np.max(idx_img, axis=0)
	y_min, y_max = np.min([y_min, y_max]), np.max([y_min, y_max])
	x_min, x_max = np.min([x_min, x_max]), np.max([x_min, x_max])

	# obj_img = img[y_min:y_max, x_min:x_max]
	idx_img = {'00':y_min, '01':y_max, '10':x_min, '11':x_max}
	# obj_img = img[idx_img['00']:idx_img['01'], idx_img['10']:idx_img['11']]
	
    	return idx_img

    def select_object_by_idx_img(img, idx_img):
	return img[idx_img['00']:idx_img['01'], idx_img['10']:idx_img['11']]

    
    def adjustCoordinates(idx_table_scene, idx_obj_table):
	idx_obj_scene = {}
	for key in idx_obj_table.keys():
	idx_obj_scene[key] = idx_obj_table[key] + idx_table_scene[key]

	return idx_obj_scene
	
    def centre_coordinate(idx_scene):
	idx = np.array([0.5*(idx_scene['00']+idx_scene['01']), 0.5*(idx_scene['10']+idx_scene['11'])])
	idx = list(map(int, idx))
	return idx
	

    def convert_idx_to_world(self, f, Depth, PCL, idx):
	cameraFrame = RealWorld.frame("camera")
	camera_position = cameraFrame.getPosition()
	camera_rotation = cameraFrame.getRotationMatrix()
	f = f * 360.
	fxfypxpy = [f, f, 320., 180.]
	Depth_img = PCL[idx[0], idx[1]]

	idx_img_camera = np.ones(4)
	idx_img_camera[:2] = Depth_img[:2]*np.array(idx, dtype='float')/f
	idx_img_camera[-2] = Depth_img[-1]

	idx_world = self.extrinsic@idx_img_camera
	return idx_world


    def give_center_by_color(self, RGB, Depth, PCL, f=0.895):

	# Cut the table
	idx_top, idx_bottom, idx_left, idx_right = self.get_table_idx(Depth)
	RGB_table = RGB[idx_top:idx_bottom,idx_left:idx_right]
	
	# Get coordinates of object by color on the table
	channel_helper = {'blue':0, 'red':2}
	idx_object_table = self.get_coordinates(RGB_table, self.color, channel_helper)

	# Table coordinates
	idx_table_scene = {}
	for key in ['00','01']:
	    idx_table_scene[key] = idx_top
	for key in ['10','11']:
	    idx_table_scene[key] = idx_left
	
	# Adjust object coordinates on the table to the image coordinates
	idx_object_scene = self.adjustCoordinates(idx_table_scene, idx_object_table)
	idx_object = self.centre_coordinate(idx_object_scene)
	obj_world = self.convert_idx_to_world(f, Depth, PCL, idx_object)
	
	return obj_world


    @staticmethod
    def _distance(input_vector: np.array, output_vector: np.array):
        return np.linalg.norm(output_vector - input_vector)

    def __call__(self, RGB: np.ndarray, Depth: np.ndarray, PCL: np.ndarray):
    	output_vector = self.give_center_by_color(RGB, Depth, f=0.895)
        pass
        
        