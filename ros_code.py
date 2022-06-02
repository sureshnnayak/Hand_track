import rospy
from std_msgs.msg import String

#======================================ROS====================================
pub = rospy.Publisher('Hand', String, queue_size=10)
rospy.init_node('talker', anonymous=True)
rate = rospy.Rate(10) # 10hz

#while True:

    #======================================ROS====================================

    