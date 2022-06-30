import argparse
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
from utils_planner import odometryCb
from gtts import gTTS
import os
import pyttsx3

engine = pyttsx3.init()
engine.setProperty('voice', 'english_rp+f3')
engine.setProperty('rate', 160) 
engine.setProperty('volume', 5) 

def text_to_speech(text):
    engine.say(text)
    engine.runAndWait()

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
    global target, hand, target_n, target_history, target_pose_list, camera
    corrected_json_string = '{' + msg.data + '}'
    data = json.loads(corrected_json_string)
    camera_pose = data['odometry']
    camera_x, camera_y, camera_z, roll, pitch, yaw = camera_pose
    if camera is None:
        camera = Hand(camera_x, camera_y, camera_z, None)
    else:
        camera.update_pose(camera_x, camera_y, camera_z, None)
    if "hand" in data:
        hand_x_C, hand_y_C, hand_z_C = data["hand"]

        if hand_x_C != 0.0 and hand_y_C != 0.0 and hand_z_C != 0.0:
            hand_x, hand_y, hand_z = camera_to_global(hand_x_C, hand_y_C, hand_z_C, camera_pose)

            if hand is None:
                print("Hand detected!")
                hand = Hand(hand_x, hand_y, hand_z, None)
            else:
                hand.update_pose(hand_x, hand_y, hand_z, None)
    elif hand is not None:
        hand.update_pose(0.0, 0.0, 0.0, None)
    if "target" in data and ready:
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
                    text_to_speech("Target located.")
                elif target_n > target_history and debug_mode:
                    target.update_pose(target_x, target_y, target_z, None)

def continuous_planner(target, hand, curr_state, prev_state, prev_command, threshold = 0.1, euc_threshold = 0.2):
    global target_n, target_pose_list, no_hand, ground, ready
    hand_in_frame_command = "Please move your hand in front of the camera."
    if not no_hand:
        if hand is None or hand.get_pose() == (0.0, 0.0, 0.0):
            command = hand_in_frame_command
            return command, curr_state, prev_state, prev_command

    if no_hand:
        diff =np.array(target.get_pose()) - np.array(camera.get_pose())
    else:
        diff = np.array(target.get_pose()) - np.array(hand.get_pose())
    print(diff)
    print("Current state: ", curr_state)
    distance = math.sqrt(diff[0]**2 + diff[1]**2 + diff[2]**2)
    if distance < euc_threshold:
        command = "Go ahead and grab the item"
        curr_state, prev_state, prev_command = None, None, None
        target_n, target_pose_list = 0, []
        ready = False
        target = None
        return command, curr_state, prev_state, prev_command

    # Calculate the difference between hand and target and assign the state accordingly
    prev_state = curr_state
    x_diff, y_diff, z_diff = diff[0], diff[1], diff[2]
    if ground:
        if abs(x_diff) > threshold:
            curr_state = "x"
            print("X state")
        elif abs(z_diff) > threshold:
            curr_state = "z"
            print("Z state")
        elif abs(y_diff) > threshold + 0.05:
            curr_state = "y"
            print("Y state")
    else:
        if abs(y_diff) > threshold + 0.05:
            curr_state = "y"
            print("Y state")
        elif abs(x_diff) > threshold:
            curr_state = "x"
            print("X state")
        elif abs(z_diff) > threshold:
            curr_state = "z"
            print("Z state")
    
    if curr_state != prev_state and prev_state is not None:
        command = "Stop"
    elif curr_state == "y":
        if y_diff < 0:
            command = "Keep on going down"
        else:
            command = "Keep on going up"
    elif curr_state == "x":
        if x_diff < 0:
            command = "Keep on going left"
        else:
            command = "Keep on going right"
    elif curr_state == "z":
        if z_diff < 0:
            command = "Keep on going forward"
        else:
            command = "Keep on going backward"

    if prev_command == command and command not in ["Stop", hand_in_frame_command]:
        command = "Keep on going"
    else:
        prev_command = command
    return command, curr_state, prev_state, prev_command


if __name__ == "__main__":
    t_265_ROS, debug_mode, ready = False, False, True
    camera, hand, target, prev_command = None, None, None, None
    target_n, target_pose_list, target_history = 0, [], 20
    curr_state, prev_state = None, None

    parser = argparse.ArgumentParser(description='Driver code for continuous planner')
    parser.add_argument('--hand', nargs='*', default=False, type=bool,
                        help='True if using hand tracking')
    parser.add_argument('--ground', nargs='*', default=False, type=bool,
                        help='True if object is on ground')
    args = parser.parse_args()
    args.hand = True if args.hand == [] else args.hand
    args.ground = True if args.ground == [] else args.ground

    no_hand = not args.hand
    ground = args.ground
    if ground: target_history = 20
    euc_threshold = 0.24
    threshold = 0.12
    # ROS node init and topic subscriptions
    rospy.init_node('continuous_planner_node')
    if t_265_ROS:
        rospy.Subscriber('/camera/odom/sample', Odometry, odometryCb)
    rospy.Subscriber('hand_and_target', String, hand_target_Cb)
    time.sleep(2)
    rate = rospy.Rate(0.01)

    while not rospy.is_shutdown():
        time.sleep(1)
        if target is None:
            continue
        print("Hand: ", hand)
        print("Target: ", target)
        print("_________________________________________")
        if target_n > target_history:
            command, curr_state, prev_state, prev_command = continuous_planner(target, hand, curr_state, prev_state, prev_command, threshold, euc_threshold)
            if prev_command is not None and "Keep on" in prev_command and "Keep on" in command:
                time.sleep(1)
            print("Command: ", command)
            print("_________________________________________")
            text_to_speech(command)
            if command == "Go ahead and grab the item":
                a = input("Input anything to continue")
                ready = True
                text_to_speech("Ready to locate another item.")