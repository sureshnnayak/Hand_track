import mediapipe as mp
import cv2
import numpy as np
import pyrealsense2 as rs

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
        color_image = cv2.flip(color_image,1)
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

            result = (int(dst[0,0,0] ), int(dst[0,0,1])), (int(dst[1,0,0] ),int(dst[1,0,1])),\
             (int(dst[2,0,0]), int(dst[2,0,1])) ,(int(dst[3,0,0]), int(dst[3,0,1]))


        return result








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

#device = connected_devices[0] # use this when we are only using one camera
device_rgb = '115422250228' #Hardcoding the device Number 
pipeline_rgb = rs.pipeline()
config_rgb = rs.config()
background_removed_color = 153 # Grey


# ====== Mediapipe ======
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils


# ====== Enable Streams ======
config_rgb.enable_device(device_rgb)


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


align_to = rs.stream.color
align = rs.align(align_to)

# ====== Get depth Scale ======
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
print("Depth:")
print(depth_scale)
print(f"\tDepth Scale for Camera SN {device_rgb} is: {depth_scale}")


file_name = 'sift_trial/ramen.jpeg'
target = Target(file_name)

while True:
        frames = pipeline_rgb.wait_for_frames()
        aligned_frames = align.process(frames)
        aligned_depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        
        if not aligned_depth_frame or not color_frame:
            continue
        target_loc= target.find_target_sift(color_frame)

        #print (target_loc)
        color_image = np.asanyarray(color_frame.get_data())
        images = color_image
        if target_loc:
            bottomLeft, topLeft, topRight,bottomRight = target_loc

            cv2.line(images, topLeft, topRight, (255, 0, 0), 2)
            cv2.line(images, topRight, bottomRight, (255, 0, 0), 2)
            cv2.line(images, bottomRight, bottomLeft, (255, 0, 0), 2)
            cv2.line(images, bottomLeft, topLeft, (255, 0, 0), 2) # Red
        cv2.namedWindow("name_of_window", cv2.WINDOW_AUTOSIZE)
        cv2.imshow("name_of_window", images)
        key = cv2.waitKey(1)
        # Press esc or 'q' to close the image window
        if key & 0xFF == ord('q') or key == 27:
            print(f"User pressed break key for SN: {device_rgb}")
            break

 
