# Hand_track

Identifying a item in a grocery store is a daunting task. for a visually impaired person this is evn more difficult. this presents us with a new computer vision problem of assisting the visually impaired person in picking up a identifying an picking up a item from the aisle.
 
The approach that is used in this project is straightforward. First the target object is identified  and also the hand position is identified. Once we have both coordinates we can use a simple navigation algorithm to guide the person, using verbal commands to grab an item from the shelf.
 
The media pipe provides a robust hand tracking model which can be leveraged to track the hand through the course of action. for finding the position of the object, Aruco markers are used(this can be later replaced with robust neural network model to identify the object)




Steps to Run:
1. run rosmaster:  roscore
2. run listner code : python listener.py
3. run object/hand tracking code : python media_pipe_dept.py
to list ros topics : rostopic list
4. record ros msg to bag:  rosbag record -O <bag file name> <topic1> <topic 2> eg rosbag record -O hand_target_track /hand_and_target /camera/odom/sample
5. To start the position tracker:roslaunch /home/smartcane/cane_ws/src/realsense-ros/realsense2_camera/launch/rs_t265.launch

src/realsense-ros/realsense2_camera/launch