import cv2

class Hand:
    def __init__(self, x, y, z, orientation):
        self.x = x
        self.y = y
        self.z = z
        self.orientation = orientation
        self.checkpoint_x = x
        self.checkpoint_y = y
        self.checkpoint_z = z
        self.orientation = orientation
        self.delta_x = 0.0
        self.delta_y = 0.0
        self.delta_z = 0.0

    def update_pose(self, x, y, z, orientation):
        self.x = x
        self.y = y
        self.z = z
        self.orientation = orientation

    def update_checkpoint(self):
        self.checkpoint_x = self.x
        self.checkpoint_y = self.y
        self.checkpoint_z = self.z

    def calculate_delta_pose(self):
        self.delta_x = self.x - self.checkpoint_x
        self.delta_y = self.y - self.checkpoint_y
        self.delta_z = self.z - self.checkpoint_z
        return {"start_pose": (self.checkpoint_x, self.checkpoint_y, self.checkpoint_z), 
                "stop_pose": (self.x, self.y, self.z)}
        

    def get_delta_pose(self):
        return (self.delta_x, self.delta_y, self.delta_z)

    def get_pose(self):
        return (self.x, self.y, self.z)

    def print_delta_pose(self):
        return "del_x={:.2f}, del_y={:.2f}, del_z={:.2f}".format(self.delta_x, self.delta_y, self.delta_z)

    def __str__(self):
        return "x={:.2f}, y={:.2f}, z={:.2f}".format(self.x, self.y, self.z)

    def print_6d_pose(self):
        return "x={:.2f}, y={:.2f}, z={:.2f}, x_t={:.2f}, y_t={:.2f}, z_t={:.2f}".format(self.x, self.y, self.z, 
        self.orientation[0], self.orientation[1], self.orientation[2])