#!/usr/bin/env python3

from __future__ import print_function

# Python Headers
import os 
import csv
import math
import numpy as np
from numpy import linalg as la
import scipy.signal as signal

# ROS Headers
import rospy

# GEM Sensor Headers
from std_msgs.msg import String, Bool, Float32, Float64, Float64MultiArray
from ackermann_msgs.msg import AckermannDriveStamped, AckermannDrive
from geometry_msgs.msg import Point


class PurePursuit(object):
    
    def __init__(self):
        
        # 0.5 - 0.1 - 0.41

        self.rate = rospy.Rate(30)

        self.look_ahead = 0.3 # 4
        self.wheelbase  = 0.325 # meters
        self.offset     = 0.015 # meters        
        
        self.ctrl_pub  = rospy.Publisher("/vesc/low_level/ackermann_cmd_mux/input/navigation", AckermannDriveStamped, queue_size=1)
        self.drive_msg = AckermannDriveStamped()
        self.drive_msg.header.frame_id = "f1tenth_control"
        self.drive_msg.drive.speed     = 1.2 # m/s, reference speed

        # self.vicon_sub = rospy.Subscriber('/car_state', Float64MultiArray, self.carstate_callback)
 
        ## TODO ## we should subscribe the waypoint in real world local coordinate here: 
        #I just use the pixel coordinate for the test
        self.lane_sub = rospy.Subscriber('/midpoint_pose', Point, self.waypoint_callback)

        
        self.x   = 0.0
        self.y   = 0.0
        self.yaw = 0.0
        self.z   = 0.0 # object detection flag
        
   
    def waypoint_callback(self, carstate_msg):
        self.x = carstate_msg.x
        self.y = carstate_msg.y
        self.yaw = np.arctan2(self.y, self.x)
        self.z = carstate_msg.z

        

    # computes the Euclidean distance between two 2D points
    def dist(self, p1, p2):
        return round(np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2), 3)

    def start_pp(self):
        
        while not rospy.is_shutdown():

            self.path_points_x = self.x
            self.path_points_y = self.y
            self.path_points_yaw_record = self.yaw

            curr_x = 0
            curr_y = 0
            curr_yaw = 0

            #use the relative distance not the waypoint
            self.distance = self.dist((self.path_points_x, self.path_points_y), (curr_x, curr_y))

            L = self.distance

            # find the curvature and the angle 
            alpha = self.path_points_yaw_record - curr_yaw

            # ----------------- tuning this part as needed -----------------
            k       = 1#0.12
            # angle_i = math.atan((k * 2 * self.wheelbase * math.sin(alpha)) / L) 
            # angle   = angle_i*2
            angle = math.atan2(k * 2.0 * self.wheelbase * math.sin(alpha),L)
            # ----------------- tuning this part as needed -----------------

            # # f_delta = round(np.clip(angle, -0.3, 0.3), 3)
            f_delta = round(angle, 3)

            f_delta_deg = round(np.degrees(angle))

            ct_error = round(np.sin(alpha) * L, 3)
            print("Crosstrack Error: " + str(ct_error))
            print("Front steering angle: " + str(f_delta_deg) + " degrees")
            print("\n")

            self.drive_msg.header.stamp = rospy.get_rostime()
            self.drive_msg.drive.steering_angle = f_delta

            #----------------------velocity control---------------------------
            flag = self.z
            if (flag == 1.0):
                self.drive_msg.drive.speed = 0
            #-----------------------------------------------------------------
            self.ctrl_pub.publish(self.drive_msg)
        
            self.rate.sleep()



def pure_pursuit():

    rospy.init_node('vicon_pp_node', anonymous=True)
    pp = PurePursuit()

    try:
        pp.start_pp()
    except rospy.ROSInterruptException:
        pass


if __name__ == '__main__':
    pure_pursuit()


