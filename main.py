import media_pipe_depth
import aruco_marker.aruco_code as aruco_code
import main


import rospy
from std_msgs.msg import String

#======================================ROS====================================
pub = rospy.Publisher('Hand', String, queue_size=10)
rospy.init_node('talker', anonymous=True)
rate = rospy.Rate(10) # 10hz


class object_track:


    def __init__(self,dev = 1):
        image_show = dev



    def hand_track():
        print("hand tracking method")

    def target_track():
        print("Target_track")
    def print_image():
        print("method to print the image")

    







def display_image(image):
    print(" ")
    images = cv2.putText(images, f"{hand_side} Hand Distance: {mfk_distance_feet:0.3} feet ({mfk_distance:0.3} m) away", org2, font, fontScale, color, thickness, cv2.LINE_AA)
    

Object_detect_count = 0
#while True:
while not rospy.is_shutdown():

    hand_info = get_hand_info()

    target_info = get_target_info()
    ros_msg = hand_info + target_info

    if key & 0xFF == ord('q') or key == 27:
        print(f"User pressed break key for SN: {device}")
        break


