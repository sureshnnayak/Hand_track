#Link to code https://medium.com/@smart-design-techology/hand-detection-in-3d-space-888433a1c1f3
# ====== Sample Code for Smart Design Technology Blog ======

# Intel Realsense D435 cam has RGB camera with 1920×1080 resolution
# Depth camera is 1280x720
# FOV is limited to 69deg x 42deg (H x V) - the RGB camera FOV

# If you run this on a non-Intel CPU, explore other options for rs.align
    # On the NVIDIA Jetson AGX we build the pyrealsense lib with CUDA

import pyrealsense2 as rs
import mediapipe as mp
import cv2
import numpy as np
import datetime as dt
import imutils
import rospy
from nav_msgs.msg import Odometry
from std_msgs.msg import String
from tf.transformations import euler_from_quaternion


import json

SIFT = 0
class Target:
    def __init__(self,target_image_location):
        self.targt_image = cv2.imread(target_image_location,cv2.IMREAD_GRAYSCALE) # queryImage

    def find_target_sift(self,color_frame):
        
        # Get and align frames
        #Local varible initialization
        THRESHOLD_MATCH_COUNT = 10

        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks = 50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        sift = cv2.SIFT_create()

        # Process images
        #depth_image = np.asanyarray(aligned_depth_frame.get_data())
        #depth_image_flipped = cv2.flip(depth_image,1)
        color_image = np.asanyarray(color_frame.get_data())

        gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
        #color_image = cv2.flip(color_image,1)
        # gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
        
        # find the keypoints and descriptors with SIFT
        kp1, des1 = sift.detectAndCompute(self.targt_image,None)
        kp2, des2 = sift.detectAndCompute(gray_image, None)

        matches = flann.knnMatch(des1,des2,k=2)

        # store all the good matches as per Lowe's ratio test.
        good = []
        for m,n in matches:
            if m.distance < 0.7*n.distance:
                good.append(m)
        result = None
        ## Eliminating not good matches by defining threshold 
        if len(good) > THRESHOLD_MATCH_COUNT:
            src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
            dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
            #matchesMask = mask.ravel().tolist()

            h,w = self.targt_image.shape
            pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
            
            dst = cv2.perspectiveTransform(pts,M)
            """ polyline_params = dict(isClosed=True,
                                color=(255,0,0),
                                thickness=10,
                                lineType=cv2.LINE_AA,
                                shift=0) """
            #img2 = cv2.polylines(img2,[np.int32(dst)],True,(0, 255 ,0),4, cv2.LINE_AA)

            #result = (int(dst[0,0,0] + self.targt_image.shape[1]), int(dst[0,0,1])), (int(dst[1,0,0] + self.targt_image.shape[1]),int(dst[1,0,1])),\
            # (int(dst[2,0,0] + self.targt_image.shape[1]), int(dst[2,0,1])) ,(int(dst[3,0,0] + self.targt_image.shape[1]), int(dst[3,0,1]))

            result = ( int(dst[0,0,0] ), int(dst[0,0,1])), (int(dst[1,0,0] ),int(dst[1,0,1])),\
             (int(dst[2,0,0]), int(dst[2,0,1])) ,( int(dst[3,0,0]), int(dst[3,0,1]))

        print(result)
        return result




    

#======================================Target Object detection================
# define names of each possible ArUco tag OpenCV supports
ARUCO_DICT = {
	"DICT_4X4_50": cv2.aruco.DICT_4X4_50,
	"DICT_4X4_100": cv2.aruco.DICT_4X4_100,
	"DICT_4X4_250": cv2.aruco.DICT_4X4_250,
	"DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
	"DICT_5X5_50": cv2.aruco.DICT_5X5_50,
	"DICT_5X5_100": cv2.aruco.DICT_5X5_100,
	"DICT_5X5_250": cv2.aruco.DICT_5X5_250,
	"DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
	"DICT_6X6_50": cv2.aruco.DICT_6X6_50,
	"DICT_6X6_100": cv2.aruco.DICT_6X6_100,
	"DICT_6X6_250": cv2.aruco.DICT_6X6_250,
	"DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
	"DICT_7X7_50": cv2.aruco.DICT_7X7_50,
	"DICT_7X7_100": cv2.aruco.DICT_7X7_100,
	"DICT_7X7_250": cv2.aruco.DICT_7X7_250,
	"DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
	"DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
	"DICT_APRILTAG_16h5": cv2.aruco.DICT_APRILTAG_16h5,
	"DICT_APRILTAG_25h9": cv2.aruco.DICT_APRILTAG_25h9,
	"DICT_APRILTAG_36h10": cv2.aruco.DICT_APRILTAG_36h10,
	"DICT_APRILTAG_36h11": cv2.aruco.DICT_APRILTAG_36h11
}
def find_target_aruco(image, type="DICT_5X5_100"):
    #resize the image
    #image = imutils.resize(image, width=600)
    # verify that the supplied ArUCo tag exists and is supported by
    # OpenCV

    # load the ArUCo dictionary, grab the ArUCo parameters, and find_target_aruco
    # the markers
    dimensions = image.shape
    arucoDict = cv2.aruco.Dictionary_get(ARUCO_DICT[type])
    arucoParams = cv2.aruco.DetectorParameters_create()
    (corners, ids, rejected) = cv2.aruco.detectMarkers(image, arucoDict,
        parameters=arucoParams)
    y_len, x_len,z_len = dimensions
    x_len -=2
    y_len -=2

    if len(corners) > 0:
        #print(len(corners))
        ids = ids.flatten()
        # loop over the detected ArUCo corners
        for (markerCorner, markerID) in zip(corners, ids):
            #=====================
            corners = markerCorner.reshape((4, 2))
            (topLeft, topRight, bottomRight, bottomLeft) = corners
            # convert each of the (x, y)-coordinate pairs to integers

            topRight = ( x_len - int(topRight[0]),  int(topRight[1]))
            bottomRight = (x_len -int(bottomRight[0]), int(bottomRight[1]))
            bottomLeft = (x_len -int(bottomLeft[0]), int(bottomLeft[1]))
            topLeft = (x_len - int(topLeft[0]), int(topLeft[1]))

            result = (topRight, bottomRight, bottomLeft,topLeft)
            return result
        #return cX,cY #only last object is returned in this case. if there are multiple objects we need to create a list 
    return None
#======================================Target Object detection================

font = cv2.FONT_HERSHEY_SIMPLEX
org = (20, 100)
fontScale = .5
color = (0,50,255)
thickness = 1

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

# ====== Realsense ======
realsense_ctx = rs.context()
connected_devices = [] # List of serial numbers for present cameras
for i in range(len(realsense_ctx.devices)):
    detected_camera = realsense_ctx.devices[i].get_info(rs.camera_info.serial_number)
    print(f"{detected_camera}")
    connected_devices.append(detected_camera)
print(connected_devices)
#device = connected_devices[0] # use this when we are only using one camera
device_rgb = '115422250228' #Hardcoding the device Number 
pipeline_rgb = rs.pipeline()
config_rgb = rs.config()
background_removed_color = 153 # Grey

device_odo = '952322110941'
pipeline_odo = rs.pipeline()
config_odo = rs.config()


# ====== Mediapipe ======
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils


# ====== Enable Streams ======
config_rgb.enable_device(device_rgb)
config_odo.enable_device(device_odo)

# # For worse FPS, but better resolution:
# stream_res_x = 1280
# stream_res_y = 720
# # For better FPS. but worse resolution:
stream_res_x = 640
stream_res_y = 480

stream_fps = 30

config_rgb.enable_stream(rs.stream.depth, stream_res_x, stream_res_y, rs.format.z16, stream_fps)
config_rgb.enable_stream(rs.stream.color, stream_res_x, stream_res_y, rs.format.bgr8, stream_fps)
profile = pipeline_rgb.start(config_rgb)

pipeline_odo.start(config_odo)
align_to = rs.stream.color
align = rs.align(align_to)

# ====== Get depth Scale ======
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
print("Depth:")
print(depth_scale)
print(f"\tDepth Scale for Camera SN {device_rgb} is: {depth_scale}")


# ====== Get and process images ====== 
print(f"Starting to capture images on SN: {device_rgb}")


#======================================ROS====================================
pub = rospy.Publisher('hand_and_target', String, queue_size=10)
rospy.init_node('talker', anonymous=True)
rate = rospy.Rate(10) # 10hz




camera = None

Object_detect_count = 0
file_name = 'sift_trial/ramen.jpeg'
target = Target(file_name)
#while True:
while not rospy.is_shutdown():

    #======================================ROS====================================
    hand_info = None
    target_info = None
    number_of_hands = 0
    start_time = dt.datetime.today().timestamp()
    
    odo_frames = pipeline_odo.wait_for_frames()
    

        # Fetch pose frame
    odo_pose = odo_frames.get_pose_frame()
    if odo_pose:
            # Print some of the pose data to the terminal
            #data = odo_pose.get_pose_data()
            #print("Frame #{}".format(odo_pose.frame_number))
            #print("Position: {}".format(data.translation))
            #print("Velocity: {}".format(data.velocity))
            data = odo_pose.get_pose_data()

            roll, pitch, yaw = euler_from_quaternion([data.rotation.x, data.rotation.y, data.rotation.z, data.rotation.w])

            odometry = [data.translation.x, data.translation.y, data.translation.z, roll, pitch, yaw]
            odometry_data = '"odometry":[{0},{1},{2},{3},{4},{5}]'.format(odometry[0],odometry[1],odometry[2],odometry[3],odometry[4],odometry[5])


    # Get and align frames
    frames = pipeline_rgb.wait_for_frames()
    aligned_frames = align.process(frames)
    aligned_depth_frame = aligned_frames.get_depth_frame()
    color_frame = aligned_frames.get_color_frame()

    #get odometry data
    #odometry_data = None
    #if camera:
        #camera_x, camera_y, camera_z = camera.get_pose()
        #roll, pitch, yaw = camera.orientation
       #odometry_data = '"odometry":[{0},{1},{2},{3},{4},{5}]'.format(odometry)
        
    if not aligned_depth_frame or not color_frame:
        continue

    # Process images
    depth_image = np.asanyarray(aligned_depth_frame.get_data())
    depth_image_flipped = cv2.flip(depth_image,1)
    color_image = np.asanyarray(color_frame.get_data())

    
    target_loc = None
    if SIFT:
        target_loc= target.find_target_sift(color_frame)
    else:
        target_loc = find_target_aruco(color_image)
    if target_loc:
        bottomLeft, topLeft, topRight,bottomRight = target_loc
    #images = color_image
    color_image_flipped = cv2.flip(color_image,1)
    images = color_image

    color_images_rgb = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)

    # Process hands
    results = hands.process(color_images_rgb)

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
            
            hand_z = depth_image_flipped[y,x] * depth_scale # meters
            #mfk_distance_feet = hand_z * 3.281 # feet
            images = cv2.putText(images, f"{hand_side} Hand Distance: {hand_z:0.3} m away", org2, font, fontScale, color, thickness, cv2.LINE_AA)
            #print("depth:", mfk_distance)
            i+=1
            
            #======================================ROS==================================== 


            if hand_side =="Right"  :
                x_meter = ((x - (REALSENSE_PPX)) / REALSENSE_FX) * hand_z #rel_positions[i][1]
                y_meter = ((y - (REALSENSE_PPY)) / REALSENSE_FY) * hand_z #rel_positions[i][1]
                
                hand_info= '"hand":[{0},{1},{2}]'.format(x_meter, y_meter, hand_z)
                
    if target_loc:#target
       
        target_x = int((topLeft[0] + bottomRight[0]) / 2.0)
        target_y = int((topLeft[1] + bottomRight[1]) / 2.0)
        target_z = depth_image_flipped[target_y,target_x] * depth_scale # meters
        #print(target_z, depth_image_flipped[target_x,target_y] * depth_scale )# meters)

        #target_z = depth_image[target_y,target_x] * depth_scale # meters
        target_x_meter = ((target_x - (REALSENSE_PPX)) / REALSENSE_FX) * target_z #rel_positions[i][1]
        target_y_meter = ((target_y - (REALSENSE_PPY)) / REALSENSE_FY) * target_z #rel_positions[i][1]
        
        target_info = '"target":[{0},{1},{2}]'.format(target_x_meter,target_y_meter,target_z)
        #target_info = json.dumps(target_info)
        #st_l = '{'
        #st_r = '}'
        #target_info = st_l + target_info + st_r
        org2 = (20, org[1]+(20*(5+1)))
        images = cv2.putText(images, f"Target Distance: {target_z:0.3} m away", org2, font, fontScale, color, thickness, cv2.LINE_AA)
        

        #"My name is {0}, I'm {1}".format("John",36)
        #hand_info = "z(depth):"+ str(mfk_distance) +",\tx:" + str(x_meter)+"\ty:" + str(y_meter) 
        #target_info = "\tTarget_x" + str(target_x_meter)+ "\ttarget_y:" + str(target_y_meter)
    ros_msg = None
    comma= ','
    if hand_info and target_info:
        ros_msg = hand_info +comma+ target_info 
       
    elif hand_info:
        ros_msg = hand_info 
        
    elif target_info:
        ros_msg =  target_info 
   
    if odometry_data and ros_msg:

        ros_msg = ros_msg + comma + odometry_data
        pub.publish(ros_msg)

            #======================================ROS====================================



    # Display FPS
    time_diff = dt.datetime.today().timestamp() - start_time
    fps = int(1 / time_diff)
    org3 = (20, org[1] + 60)
    images = cv2.putText(images, f"FPS: {fps}", org3, font, fontScale, color, thickness, cv2.LINE_AA)

    name_of_window = 'SN: ' + str(device_rgb)

    # ====================== adding Bounding Box to target image==================================================
    if target_loc:
        if Object_detect_count < 100 :
            Object_detect_count = 0
        else :
            Object_detect_count +=1
        # draw the bounding box of the ArUCo detection
        cv2.line(images, topLeft, topRight, (255, 0, 0), 2)
        cv2.line(images, topRight, bottomRight, (255, 0, 0), 2)
        cv2.line(images, bottomRight, bottomLeft, (255, 0, 0), 2)
        cv2.line(images, bottomLeft, topLeft, (255, 0, 0), 2) # Red 
    
    # show the output image
    # Display images 
    cv2.namedWindow(name_of_window, cv2.WINDOW_AUTOSIZE)
    cv2.imshow(name_of_window, images)
    key = cv2.waitKey(1)
    # Press esc or 'q' to close the image window
    if key & 0xFF == ord('q') or key == 27:
        print(f"User pressed break key for SN: {device_rgb}")
        break
    
print(f"Application Closing")
pipeline_rgb.stop()
print(f"Application Closed.")


#if __name__ == "__main__":
    
