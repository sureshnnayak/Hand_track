import pyrealsense2 as rs
import mediapipe as mp
import cv2
import numpy as np
import datetime as dt
import imutils
import rospy
from std_msgs.msg import String


#=================REALSENSE======================

REALSENSE_FOCAL_LENGTH = 1.93  # millimeters
REALSENSE_SENSOR_HEIGHT_MM = 2.454  # millimeters
REALSENSE_SENSOR_WIDTH_MM = 3.896  # millimeters
REALSENSE_REZ_HEIGHT = 800  # pixels
REALSENSE_REZ_WIDTH = 1280  # pixels
REALSENSE_FX = 628.071 # D455
REALSENSE_PPX = 637.01 # D455
# REALSENSE_FX = 616.6 # D435
# REALSENSE_PPX = 318.5 # D435
REALSENSE_FX = 383.0088195800781 # D455
REALSENSE_PPX = 320.8406066894531 # D455
#ppy is:  238.125
REALSENSE_PPY = 238.125 #D455

#fy is:  383.0088195800781
REALSENSE_FY = 383.0088195800781 #D455
#=================REALSENSE======================


class Hand:
    def __init__(self):
        font = cv2.FONT_HERSHEY_SIMPLEX
        org = (20, 100)
        fontScale = .5
        color = (0,50,255)
        thickness = 1

font = cv2.FONT_HERSHEY_SIMPLEX
org = (20, 100)
fontScale = .5
color = (0,50,255)
thickness = 1






# ====== Realsense ======
realsense_ctx = rs.context()
connected_devices = [] # List of serial numbers for present cameras
for i in range(len(realsense_ctx.devices)):
    detected_camera = realsense_ctx.devices[i].get_info(rs.camera_info.serial_number)
    print(f"{detected_camera}")
    connected_devices.append(detected_camera)
device = connected_devices[0] # In this example we are only using one camera
pipeline = rs.pipeline()
config = rs.config()
background_removed_color = 153 # Grey

# ====== Mediapipe ======
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils


# ====== Enable Streams ======
config.enable_device(device)

# # For worse FPS, but better resolution:
# stream_res_x = 1280
# stream_res_y = 720
# # For better FPS. but worse resolution:
stream_res_x = 640
stream_res_y = 480

stream_fps = 30

config.enable_stream(rs.stream.depth, stream_res_x, stream_res_y, rs.format.z16, stream_fps)
config.enable_stream(rs.stream.color, stream_res_x, stream_res_y, rs.format.bgr8, stream_fps)
profile = pipeline.start(config)

align_to = rs.stream.color
align = rs.align(align_to)

# ====== Get depth Scale ======
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
print(f"\tDepth Scale for Camera SN {device} is: {depth_scale}")

# ====== Set clipping distance ======
clipping_distance_in_meters = 2
clipping_distance = clipping_distance_in_meters / depth_scale
print(f"\tConfiguration Successful for SN {device}")

# ====== Get and process images ====== 
print(f"Starting to capture images on SN: {device}")


def tack_hand():

    start_time = dt.datetime.today().timestamp()

    # Get and align frames
    frames = pipeline.wait_for_frames()
    aligned_frames = align.process(frames)
    aligned_depth_frame = aligned_frames.get_depth_frame()
    color_frame = aligned_frames.get_color_frame()
    
    if not aligned_depth_frame or not color_frame:
        return

    # Process images
    depth_image = np.asanyarray(aligned_depth_frame.get_data())
    depth_image_flipped = cv2.flip(depth_image,1)
    color_image = np.asanyarray(color_frame.get_data())


    #topRight, bottomRight, bottomLeft,topLeft = detect( cv2.flip(color_image,1))
    
    depth_image_3d = np.dstack((depth_image,depth_image,depth_image)) #Depth image is 1 channel, while color image is 3
    background_removed = np.where((depth_image_3d > clipping_distance) | (depth_image_3d <= 0), background_removed_color, color_image)

    depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
    images = background_removed
    #images = cv2.flip(background_removed,1)
    #color_image = cv2.flip(color_image,1)
    

    

    color_images_rgb = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)

    # Process hands
    results = hands.process(color_images_rgb)
    #=============================================target detection======================================
    #topRight, bottomRight, bottomLeft,topLeft = detect(color_images_rgb)
    
    #target_x, target_y = detect(color_images_rgb)

    
    if results.multi_hand_landmarks:
        number_of_hands = len(results.multi_hand_landmarks)
        i=0
        for handLms in results.multi_hand_landmarks:
            mpDraw.draw_landmarks(images, handLms, mpHands.HAND_CONNECTIONS)
            org2 = (20, org[1]+(20*(i+1)))
            hand_side_classification_list = results.multi_handedness[i]
            hand_side = hand_side_classification_list.classification[0].label
            middle_finger_knuckle = results.multi_hand_landmarks[i].landmark[9]
            x = int(middle_finger_knuckle.x*len(depth_image_flipped[0]))
            y = int(middle_finger_knuckle.y*len(depth_image_flipped))
            if x >= len(depth_image_flipped[0]):
                x = len(depth_image_flipped[0]) - 1
            if y >= len(depth_image_flipped):
                y = len(depth_image_flipped) - 1
            mfk_distance = depth_image_flipped[y,x] * depth_scale # meters
            mfk_distance_feet = mfk_distance * 3.281 # feet
            images = cv2.putText(images, f"{hand_side} Hand Distance: {mfk_distance_feet:0.3} feet ({mfk_distance:0.3} m) away", org2, font, fontScale, color, thickness, cv2.LINE_AA)
            #print("depth:", mfk_distance)
            i+=1
            
            #======================================ROS==================================== 
            if hand_side =="Left" and topLeft != -1 :
           
                target_x = int((topLeft[0] + bottomRight[0]) / 2.0)
                target_y = int((topLeft[1] + bottomRight[1]) / 2.0)
                target_x_meter = ((target_x - (REALSENSE_PPX)) / REALSENSE_FX) * mfk_distance #rel_positions[i][1]
                target_y_meter = ((target_y - (REALSENSE_PPY)) / REALSENSE_FY) * mfk_distance #rel_positions[i][1]
                x_meter = ((x - (REALSENSE_PPX)) / REALSENSE_FX) * mfk_distance #rel_positions[i][1]
                y_meter = ((y - (REALSENSE_PPY)) / REALSENSE_FY) * mfk_distance #rel_positions[i][1]

                hand_info= "'hand':({0},{1},{2}),".format(x_meter, y_meter, mfk_distance)
                target_info = "'Target':({0},{1},{2})".format(target_x_meter,target_y_meter,0)
                ros_msg = hand_info + target_info

                
                pub.publish(ros_msg)
            #======================================ROS====================================

        images = cv2.putText(images, f"Hands: {number_of_hands}", org, font, fontScale, color, thickness, cv2.LINE_AA)
    else:
        images = cv2.putText(images,"No Hands", org, font, fontScale, color, thickness, cv2.LINE_AA)


    # Display FPS
    time_diff = dt.datetime.today().timestamp() - start_time
    fps = int(1 / time_diff)
    org3 = (20, org[1] + 60)
    images = cv2.putText(images, f"FPS: {fps}", org3, font, fontScale, color, thickness, cv2.LINE_AA)

    name_of_window = 'SN: ' + str(device)

    # ====================== adding Bounding Box to target image==================================================
    if topLeft != -1:
        if Object_detect_count < 100 :
            Object_detect_count = 0
        else :
            Object_detect_count +=1
        # draw the bounding box of the ArUCo detection
        cv2.line(images, topLeft, topRight, (0, 255, 0), 2)
        cv2.line(images, topRight, bottomRight, (0, 255, 0), 2)
        cv2.line(images, bottomRight, bottomLeft, (0, 255, 0), 2)
        cv2.line(images, bottomLeft, topLeft, (0, 255, 0), 2)
    
    # show the output image
    # Display images 
    cv2.namedWindow(name_of_window, cv2.WINDOW_AUTOSIZE)
    cv2.imshow(name_of_window, images)
    key = cv2.waitKey(1)
    # Press esc or 'q' to close the image window
    