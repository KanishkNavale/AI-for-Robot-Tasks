import sys
sys.path.append('../../build')

import libry as ry
import os, copy

import cv2 as cv
import numpy as np

import matplotlib.pyplot as plt


RealWorld = ry.Config()
RealWorld.addFile(os.path.abspath('robot_scene/scene_camera.g'))
S = RealWorld.simulation(ry.SimulatorEngine.bullet, True)
S.addSensor("camera")


S.selectSensor("camera")
cameraFrame = RealWorld.frame("camera")
[rgb, depth] = S.getImageAndDepth()


def get_table_idx(depth):
	idx_list = np.arange(depth.shape[1])[depth[225,:]==np.min(depth[225,:])]
	idx_left, idx_right = idx_list[0],idx_list[-1]
	depth_table_left = depth[:,idx_left]
	idx_list_2 = np.arange(depth.shape[0])[depth_table_left==np.min(depth_table_left)]
	depth_left_diff = depth_table_left[1:] - depth_table_left[:-1]
	idx_top, idx_bottom = np.argmin(depth_left_diff), np.argmax(depth_left_diff)

	return idx_top, idx_bottom, idx_left, idx_right

idx_top, idx_bottom, idx_left, idx_right = get_table_idx(depth)
depth_table = depth[idx_top:idx_bottom,idx_left:idx_right]
rgb_table = rgb[idx_top:idx_bottom,idx_left:idx_right]


channel_helper = {'blue':0, 'red':2}
def get_coordinates(img, color, channel_helper):
    result = img.copy()
    lower = np.array([0])
    upper = np.array([25])
    channel = channel_helper[color]
    mask = cv.inRange(img[:,:,channel], lower, upper)
    idx_img = np.argwhere(mask==255)
    point_min, point_max = np.min(idx_img, axis=0), np.max(idx_img, axis=0)
    (y_min, x_min), (y_max, x_max) = np.min(idx_img, axis=0), np.max(idx_img, axis=0)
    y_min, y_max = np.min([y_min, y_max]), np.max([y_min, y_max])
    x_min, x_max = np.min([x_min, x_max]), np.max([x_min, x_max])
    
    # obj_img = img[y_min:y_max, x_min:x_max]
    idx_img = {'00':y_min, '01':y_max, '10':x_min, '11':x_max}
    # obj_img = img[idx_img['00']:idx_img['01'], idx_img['10']:idx_img['11']]
    return idx_img


def select_object_by_idx_img(img, idx_img):
    return img[idx_img['00']:idx_img['01'], idx_img['10']:idx_img['11']]


idx_blue_table = get_coordinates(rgb_table, 'blue', channel_helper)
obj_blue = select_object_by_idx_img(rgb_table, idx_blue_table)


idx_red_table = get_coordinates(rgb_table, 'red', channel_helper)
obj_red = select_object_by_idx_img(rgb_table, idx_red_table)

# Table coordinates
idx_table_scene = {}
for key in ['00','01']:
    idx_table_scene[key] = idx_top
for key in ['10','11']:
    idx_table_scene[key] = idx_left


def adjustCoordinates(idx_table_scene, idx_obj_table):
    idx_obj_scene = {}
    for key in idx_obj_table.keys():
        idx_obj_scene[key] = idx_obj_table[key] + idx_table_scene[key]
    
    return idx_obj_scene



idx_red_scene = adjustCoordinates(idx_table_scene, idx_red_table)
obj_red_scene = select_object_by_idx_img(rgb, idx_red_scene)

idx_blue_scene = adjustCoordinates(idx_table_scene, idx_blue_table)
obj_blue_scene = select_object_by_idx_img(rgb, idx_blue_scene)



def centre_coordinate(idx_scene):
    idx = np.array([0.5*(idx_scene['00']+idx_scene['01']), 0.5*(idx_scene['10']+idx_scene['11'])])
    idx = list(map(int, idx))
    return idx

### Converting from image coordinates to the camera coordinates
def convert_idx_to_world(RealWorld, S, f, depth, idx):
    cameraFrame = RealWorld.frame("camera")
    camera_position = cameraFrame.getPosition()
    camera_rotation = cameraFrame.getRotationMatrix()
    f = f * 360.
    fxfypxpy = [f, f, 320., 180.]
    points = S.depthData2pointCloud(depth, fxfypxpy)
    depth_img = points[idx[0], idx[1]]
    
    idx_img_camera = np.zeros(camera_position.shape)
    idx_img_camera[:2] = depth_img[:2]*np.array(idx, dtype='float')/f
    idx_img_camera[-1] = depth_img[-1]

    idx_world = camera_position + camera_rotation@idx_img_camera
    return idx_world


obj0 = RealWorld.getFrame("obj0")
idx_red = centre_coordinate(idx_red_scene)
obj0_world = convert_idx_to_world(RealWorld, S, 0.895, depth, idx_red)
print('obj0_world:',obj0_world)
print('obj0.getPosition():',obj0.getPosition())


obj1 = RealWorld.getFrame("obj1")
idx_blue = centre_coordinate(idx_blue_scene)
obj1_world = convert_idx_to_world(RealWorld, S, 0.895, depth, idx_blue)
print('obj1_world:',obj1_world)
print('obj1.getPosition():',obj1.getPosition())


plt.imshow(rgb)
plt.show()


def give_center_by_color(color, RealWorld, S, f=0.895):
	S.selectSensor("camera")
	cameraFrame = RealWorld.frame("camera")
	[rgb, depth] = S.getImageAndDepth()
	
	# Cut the table
	idx_top, idx_bottom, idx_left, idx_right = get_table_idx(depth)
	rgb_table = rgb[idx_top:idx_bottom,idx_left:idx_right]
	
	# Get coordinates of object by color on the table
	channel_helper = {'blue':0, 'red':2}
	idx_object_table = get_coordinates(rgb_table, color, channel_helper)

	# Table coordinates
	idx_table_scene = {}
	for key in ['00','01']:
	    idx_table_scene[key] = idx_top
	for key in ['10','11']:
	    idx_table_scene[key] = idx_left
	
	# Adjust object coordinates on the table to the image coordinates
	idx_object_scene = adjustCoordinates(idx_table_scene, idx_object_table)
	idx_object = centre_coordinate(idx_object_scene)
	obj_world = convert_idx_to_world(RealWorld, S, f, depth, idx_object)
	
	return obj_world
	

obj0_world = give_center_by_color('red', RealWorld, S)
print('obj0_world:',obj0_world)
print('obj0.getPosition():',obj0.getPosition())


obj1_world = give_center_by_color('blue', RealWorld, S)
print('obj1_world:',obj1_world)
print('obj1.getPosition():',obj1.getPosition())
