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

from pyparsing import null_debug_action
from pytesseract import Output
import pytesseract
import argparse
import cv2
from PIL import Image




class Realsense :
    def __init__(self): 
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

        device_rgb = '115422250228' #Hardcoding the device Number 
        self.pipeline_rgb = rs.pipeline()
        config_rgb = rs.config()
        background_removed_color = 153 # Grey

        device_odo = '952322110941'
        self.pipeline_odo = rs.pipeline()
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
        profile = self.pipeline_rgb.start(config_rgb)

        self.pipeline_odo.start(config_odo)
        align_to = rs.stream.color
        self.align = rs.align(align_to)

        # ====== Get depth Scale ======
        depth_sensor = profile.get_device().first_depth_sensor()
        depth_scale = depth_sensor.get_depth_scale()
        print("Depth:")
        print(depth_scale)
        print(f"\tDepth Scale for Camera SN {device_rgb} is: {depth_scale}")

        # ====== Get and process images ====== 
        print(f"Starting to capture images on SN: {device_rgb}")

    def __del__(self):
        self.pipeline_rgb.stop()


    def get_image(self):
        if not rospy.is_shutdown():
            start_time = dt.datetime.today().timestamp()
            
            odo_frames = self.pipeline_odo.wait_for_frames()
            

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


            # Get and align frames
            frames = self.pipeline_rgb.wait_for_frames()
            aligned_frames = self.align.process(frames)
            aligned_depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()
  
            if not aligned_depth_frame or not color_frame:
                return
                #continue

            # Process images
            self.depth_image = np.asanyarray(aligned_depth_frame.get_data())
            self.color_image = np.asanyarray(color_frame.get_data())
            #self.show_img()

    def show_img(self,image=None):

        cv2.namedWindow("realsense", cv2.WINDOW_AUTOSIZE)
        if image is not None:
            cv2.imshow("Image capture from realsense", image)
        else:
            cv2.imshow("Image capture from realsense", self.color_image)

        cv2.waitKey(1)

        
    

r = Realsense()
while True:
    #r.get_image()
    #r.show_img()
    #image = r.color_image
    #image = cv2.imread("dunkin.jpeg")
    image = cv2.imread("apple_support_logo.png")

    #image = cv2.imread("ramen.jpeg")
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pytesseract.image_to_data(rgb, output_type=Output.DICT)
    print(results)
    img1 = np.array(Image.open("apple_support_logo.png"))
    #img1 = np.array(Image.open("dunkin.jpeg"))
    #img1 = np.array(Image.open("ramen.jpeg"))
    text = pytesseract.image_to_string(img1)

    print("test from OCR {}".format(text))
    print(text)



    # loop over each of the individual text localizations
    for i in range(0, len(results["text"])):
        # extract the bounding box coordinates of the text region from
        # the current result
        x = results["left"][i]
        y = results["top"][i]
        w = results["width"][i]
        h = results["height"][i]
        # extract the OCR text itself along with the confidence of the
        # text localization
        text = results["text"][i]
        #conf = 90
        conf = int(float(results["conf"][i]))
        #print("result :" + results["conf"][i])

        # filter out weak confidence text localizations
        min_conf = 50
        print(conf)
        if conf > min_conf : #and len(text) >1:
            # display the confidence and text to our terminal
            print("Confidence: {}".format(conf))
            print("Text: {}".format(text))
            print("Text len {}:".format(len(text)) )
            print("")
            # strip out non-ASCII text so we can draw the text on the image
            # using OpenCV, then draw a bounding box around the text along
            # with the text itself
            text = "".join([c if ord(c) < 128 else "" for c in text]).strip()
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(image, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                1.2, (0, 0, 255), 3)
    r.show_img(image)
    cv2.waitKey(2000)
    break