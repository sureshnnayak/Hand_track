from copy import deepcopy
import math
import time
import numpy as np
import rospy
import json
from nav_msgs.msg import Odometry
from std_msgs.msg import String
from tf.transformations import euler_from_quaternion
from hand import Hand
from gtts import gTTS
import os

def odometryCb(msg):
    global camera
    x = msg.pose.pose.position.x
    y = msg.pose.pose.position.y
    z = msg.pose.pose.position.z
    quaternion = msg.pose.pose.orientation
    orientation_list = [quaternion.x, quaternion.y, quaternion.z, quaternion.w]
    orientation = euler_from_quaternion(orientation_list)
    if camera is None:
        print("Camera located!")
        camera = Hand(x, y, z, orientation)
        # print(camera)
    else:
        camera.update_pose(x, y, z, orientation)

def text_to_speech(text):
    language = 'en'
    speech = gTTS(text=text, lang=language, slow=False)
    speech.save('text.mp3')
    os.system('DISPLAY=:0 mpg123 text.mp3')

def camera_to_global(x, y, z, camera_pose):
    global t_265_ROS, debug_mode
    if t_265_ROS:
        object_x_C, object_y_C, object_z_C = z, -x, y
    else:
        object_x_C, object_y_C, object_z_C = -x, -y, -z
    camera_x, camera_y, camera_z, roll, pitch, yaw = camera_pose
    mat_Z = np.array([[math.cos(yaw), -math.sin(yaw), 0.0], [math.sin(yaw), math.cos(yaw), 0.0], [0.0, 0.0, 1.0]])
    mat_Y = np.array([[math.cos(pitch), 0.0, math.sin(pitch)], [0.0, 1.0, 0.0], [-math.sin(pitch), 0.0, math.cos(pitch)]])
    mat_X = np.array([[1.0, 0.0, 0.0], [0.0, math.cos(roll), -math.sin(roll)], [0.0, math.sin(roll), math.cos(roll)]])
    rotation_matrix_C_W = np.matmul(np.matmul(mat_Z, mat_Y), mat_X)
    object_pose_W = np.matmul(rotation_matrix_C_W, np.array([object_x_C, object_y_C, object_z_C])) + np.array([camera_x, camera_y, camera_z])
    if debug_mode:
        print("_"*50)
        print("Object in Camera Frame: ", object_x_C, object_y_C, object_z_C)
        print("Camera in Global Frame: ", camera_x, camera_y, camera_z)
        print("Object in Global Frame: ", object_pose_W)
    return (object_pose_W[0], object_pose_W[1], object_pose_W[2])


def hand_target_Cb(msg):
    global target, hand, target_n, target_history, target_pose_list
    corrected_json_string = '{' + msg.data + '}'
    data = json.loads(corrected_json_string)
    camera_pose = data['odometry']
    if "hand" in data:
        hand_x_C, hand_y_C, hand_z_C = data["hand"]

        if hand_x_C != 0.0 and hand_y_C != 0.0 and hand_z_C != 0.0:
            hand_x, hand_y, hand_z = camera_to_global(hand_x_C, hand_y_C, hand_z_C, camera_pose)

            if hand is None:
                print("Hand detected!")
                hand = Hand(hand_x, hand_y, hand_z, None)
            else:
                hand.update_pose(hand_x, hand_y, hand_z, None)
    if "target" in data:
        target_n += 1
        if target_n <= target_history:
            target_x_C, target_y_C, target_z_C = data["target"]

            if target_x_C != 0.0 and target_y_C != 0.0 and target_z_C != 0.0:
                target_x, target_y, target_z = camera_to_global(target_x_C, target_y_C, target_z_C, camera_pose)
                target_pose_list.append((target_x, target_y, target_z))

                if target is None:
                    print("Target detected!")
                    target = Hand(target_x, target_y, target_z, None)
                elif target_n == target_history:
                    mean_target_x, mean_target_y, mean_target_z = np.mean(np.array(target_pose_list), axis=0)
                    target.update_pose(mean_target_x, mean_target_y, mean_target_z, None)
                elif target_n > target_history and debug_mode:
                    target.update_pose(target_x, target_y, target_z, None)