import numpy as np
import cv2

from pykinect_azure.k4abt.joint2d import Joint2d
from pykinect_azure.k4abt._k4abtTypes import K4ABT_JOINT_COUNT, K4ABT_SEGMENT_PAIRS
from pykinect_azure.k4abt._k4abtTypes import k4abt_skeleton2D_t, k4abt_body2D_t, body_colors
from pykinect_azure.k4a._k4atypes import K4A_CALIBRATION_TYPE_DEPTH

class Body2d:
	def __init__(self, body2d_handle):

		if body2d_handle:
			self._handle = body2d_handle
			self.id = body2d_handle.id
			self.initialize_skeleton()

	def __del__(self):

		self.destroy()

	def is_valid(self):
		return self._handle

	def handle(self):
		return self._handle

	def destroy(self):
		if self.is_valid():
			self._handle = None

	def initialize_skeleton(self):
		joints = np.ndarray((K4ABT_JOINT_COUNT,),dtype=np.object)

		for i in range(K4ABT_JOINT_COUNT):
			joints[i] = Joint2d(self._handle.skeleton.joints2D[i], i)

		self.joints = joints

	def draw(self, image, only_segments = False):

		color = (int (body_colors[self.id][0]), int (body_colors[self.id][1]), int (body_colors[self.id][2]))

		for segmentId in range(len(K4ABT_SEGMENT_PAIRS)):
			segment_pair = K4ABT_SEGMENT_PAIRS[segmentId]
			# for idx,i in enumerate(segment_pair):
			# 	print(idx,"th segment:",i)
			point1 = self.joints[segment_pair[0]].get_coordinates()
			point2 = self.joints[segment_pair[1]].get_coordinates()

			if (point1[0] == 0 and point1[1] == 0) or (point2[0] == 0 and point2[1] == 0):
				continue
			image = cv2.line(image, point1, point2,color, 2)

		if only_segments:
			return image

		for joint in self.joints:
			image = cv2.circle(image, joint.get_coordinates(), 3, color, 3)

		return image

	def get_body2d_data(self):
		data = {}
		data["id"] = self.id
		"""
		segmentId	Joint	Parent_Joint
			30 	  Eye_right 	Head
			29 	   Eye_Left 	Head
			25 	  Foot_Right Ankle_Right
			21	  Foot_Left  Ankle_Left

		"""
		#Head		
		eye_right = self.get_xy_from_segmentId(30)
		eye_left = self.get_xy_from_segmentId(29)
		data["eye"] = [(eye_right[0]+eye_left[0])//2,(eye_right[1]+eye_left[1])//2]
		
		#Foot		
		foot_right = self.get_xy_from_segmentId(25)
		foot_left = self.get_xy_from_segmentId(21)
		data["foot"] = [(foot_left[0]+foot_right[0])//2,(foot_left[1]+foot_right[1])//2]

		return data
		
	def get_xy_from_segmentId(self,segmentId):
		#In this case, the 0th pair gives the required coordinates
		segment_pair = K4ABT_SEGMENT_PAIRS[segmentId]
		coord = self.joints[segment_pair[0]].get_coordinates()
		#print(coord)
		return coord


	@staticmethod
	def create(body_handle, calibration, bodyIdx, dest_camera):

		skeleton2d_handle = k4abt_skeleton2D_t()
		body2d_handle = k4abt_body2D_t()

		for jointID,joint in enumerate(body_handle.skeleton.joints): 
			skeleton2d_handle.joints2D[jointID].position = calibration.convert_3d_to_2d(joint.position, K4A_CALIBRATION_TYPE_DEPTH, dest_camera)
			skeleton2d_handle.joints2D[jointID].confidence_level = joint.confidence_level

		body2d_handle.skeleton = skeleton2d_handle
		body2d_handle.id = bodyIdx

		return Body2d(body2d_handle)


	def __str__(self):
		"""Print the current settings and a short explanation"""
		message = f"Body Id: {self.id}\n\n"
		print(message)
		for joint in self.joints:
			message += str(joint)

		return message

