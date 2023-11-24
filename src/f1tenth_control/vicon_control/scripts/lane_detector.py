#! /usr/bin/env python
from __future__ import print_function
import sys
import copy
import time
import rospy
import math
import tf
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from std_msgs.msg import String, Float64
from geometry_msgs.msg import Point
from sensor_msgs.msg import Image, CameraInfo, CompressedImage
from cv_bridge import CvBridge, CvBridgeError
import message_filters
from collections import deque
from scipy import stats
from sensor_msgs.msg import Image
import ast
import sys
#----------------------------------------------------------------------
# Super parameter
# Path to the scripts
souce_path = "./src/f1tenth-sim/scripts/"
# souce_path = "./src/vehicle_drivers/gem_vision/gem_vision/camera_vision/scripts/"
# Activate the function
object_detection = True
lane_detection = True
#----------------------------------------------------------------------
sys.path.append(souce_path)
sys.path.append(souce_path + "Detector/")
from yolo_detect_image import yolo_detect_image



class ImageConverter:
    def __init__(self):
        self.node_name = "lane_detector"
        rospy.init_node(self.node_name)
        rospy.on_shutdown(self.cleanup)
        self.bridge = CvBridge()

        # Subscribe camera rgb and depth information
        # depth_img_topic = rospy.get_param('depth_info_topic','/zed2/zed_node/depth/depth_registered')
        # self.depth_img_sub = message_filters.Subscriber(depth_img_topic,Image)
        self.subcriber_rgb = message_filters.Subscriber('/D435I/color/image_raw', Image)
        # self.subcriber_rgb_camera = message_filters.Subscriber('/zed2/zed_node/rgb_raw/camera_info', CameraInfo)
        sync = message_filters.ApproximateTimeSynchronizer([self.subcriber_rgb], 10, 1)
        sync.registerCallback(self.multi_callback)
        # Publish Boudingbox information of objects
        self.midpoint_img_pub = rospy.Publisher("/midpoint_img", Image, queue_size=1)
        self.midpoint_pose_pub = rospy.Publisher("/midpoint_pose", Point, queue_size=1)
        self.object_detector_pub = rospy.Publisher("/object_detector", Image, queue_size=1)

    def multi_callback(self, rgb):
        try:
            rgb_frame = self.bridge.imgmsg_to_cv2(rgb, "bgr8")
        except CvBridgeError as e:
            rospy.logerr("CvBridge Error: {0}".format(e))
        point_msg = Point()
        # ----------------------------------------------------------------------
        # Main
        object_frame = rgb_frame
        detected = 0
        if object_detection == True:
            object_list, object_frame = yolo_detect_image(rgb_frame, souce_path)
            if object_list == []:
                detected = 0
            else:
                detected = 1
        if lane_detection == True:
            midpoint_list, midpoint_img = self.simple_lane_detector(rgb_frame)
        # ----------------------------------------------------------------------
        # Publisher
        # Publish the detected object
        object_img = CvBridge().cv2_to_imgmsg(object_frame, "bgr8")
        self.object_detector_pub.publish(object_img)

        # Publish the image
        ros_img = CvBridge().cv2_to_imgmsg(midpoint_img, "bgr8")
        self.midpoint_img_pub.publish(ros_img)

        # Publish the midpoint(closest)
        midpoint_x = midpoint_list[0][0]
        midpoint_y = midpoint_list[0][1]
        point_msg = Point()
        # X, Y, Z = self.Trans_Pix2Camera(midpoint_x, midpoint_y)
        ##TODO##
        #need to transform the camera origin to the car center
        point_msg.x = midpoint_x
        point_msg.y = midpoint_y
        point_msg.z = detected
        self.midpoint_pose_pub.publish(point_msg)


    def simple_lane_detector(self, frame):
        mask_list = self.create_mask(frame)
        # Thresh the lane through only color
        lane = self.thresh_lane(frame)
        # Select the region of interest
        segmented_image, mask = self.region_of_interest(lane, mask_list)
        # Find the midpoint
        midpoint, midpoint_img = self.find_lane_midpoint(segmented_image)
        # cv2.imshow("image", mask)
        # cv2.imshow("ColorOutput", midpoint_img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        return midpoint, midpoint_img

    # The mask coordinates of the region of interest, which shoule be adjusted according to the application 
    def get_vertices_for_img(self, img, mask_list):
        bottom = mask_list[0]
        top = mask_list[1]
        left_b = mask_list[2]
        right_b = mask_list[3]
        left_t = mask_list[4]
        right_t = mask_list[5] 
        # set 4 position of mask
        vert = None
        region_bottom_left = (left_b , bottom)
        region_bottom_right = (right_b, bottom)
        region_top_left = (left_t, top)
        region_top_right = (right_t, top)
        vert = np.array([[region_bottom_left , region_top_left, region_top_right, region_bottom_right]], dtype=np.int32)
        return vert
    
    def region_of_interest(self, img, mask_list):
        #defining a blank mask to start with
        mask = np.zeros_like(img)        
        #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
        if len(img.shape) > 2:
            channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
            ignore_mask_color = (255,) * channel_count
        else:
            ignore_mask_color = 255 
        vert = self.get_vertices_for_img(img, mask_list)    
        #filling pixels inside the polygon defined by "vertices" with the fill color    
        cv2.fillPoly(mask, vert, ignore_mask_color)
        #returning the image only where mask pixels are nonzero
        masked_image = cv2.bitwise_and(img, mask)
        return masked_image, mask

    def create_mask(self, frame):
        height = frame.shape[0]
        width = frame.shape[1]
        bottom = height
        top = height * 4 / 7
        left_b = 0 
        left_t = 0 
        right_b = width 
        right_t = width 
        mask_list = [bottom, top, left_b, right_b, left_t, right_t]
        return mask_list
    
    def find_lane_midpoint(self, lane_mask):
        # List to hold midpoints
        midpoints = []
        confidences = []

        # Get the height of the mask
        height = lane_mask.shape[0]

        # Define the number of horizontal slices
        num_slices = 10  # or however many slices you want to divide your image into

        # Calculate the increment for each horizontal slice
        increment = height // num_slices

        # Iterate over each horizontal slice
        for i in range(num_slices):
            # Calculate the y-coordinate of this slice
            y_coord = i * increment

            # Find all non-zero x-coordinates at this y-coordinate
            white_pixels = np.where(lane_mask[y_coord, :] > 0)[0]

            if len(white_pixels) > 0:
                # Calculate the midpoint for this slice
                leftmost = white_pixels[0]
                rightmost = white_pixels[-1]
                midpoint = ((leftmost + rightmost) // 2, y_coord)

                # Add the midpoint to the list of midpoints
                midpoints.append(midpoint)
                confidences.append(len(white_pixels))

        # # Pair each midpoint with its corresponding confidence
        # midpoint_confidence_pairs = list(zip(midpoints, confidences))

        # # Sort the pairs based on confidence, with the highest confidence first
        # midpoint_confidence_pairs.sort(key=lambda x: x[1], reverse=True)

        # # Assuming you want the single midpoint with the highest confidence
        # highest_confidence_midpoint = midpoint_confidence_pairs[0][0]

        # Optional: Draw the midpoints on the mask for visualization
        midpoint_visualized = cv2.cvtColor(lane_mask, cv2.COLOR_GRAY2BGR)

        for point in midpoints:
            cv2.circle(midpoint_visualized, point, 5, (0, 0, 255), -1)

        return midpoints, midpoint_visualized

    
    def thresh_lane(self, img):

        hsl_img = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)

        # lower_bound = np.array([0, 20 , 20])
        # upper_bound = np.array([50, 255,255])
        lower_bound = np.array([15, 100, 100]) 
        upper_bound = np.array([35, 255, 255]) 

        ColorOutput1 = cv2.inRange(hsl_img, lower_bound, upper_bound)

        hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lower_bound = np.array([20, 100, 100])  
        upper_bound = np.array([30, 255, 255]) 
        ColorOutput2 = cv2.inRange(hsv_img, lower_bound, upper_bound)

        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        lower_bound = np.array([150, 150, 0], dtype="uint8")  
        upper_bound = np.array([255, 255, 120], dtype="uint8")
        ColorOutput3 = cv2.inRange(rgb_img, lower_bound, upper_bound)

        canny_img = cv2.Canny(img, 50, 150)

        combined_mask = cv2.bitwise_or(ColorOutput1, ColorOutput2)
        combined_mask = cv2.bitwise_or(combined_mask, ColorOutput3)
        # combined_mask = cv2.bitwise_or(combined_mask, canny_img)

        return combined_mask
    
    def perspective_transform(self, img, verbose=False):
        
        img = img.astype('uint8') * 255
        
        IMAGE_H = img.shape[0]
        IMAGE_W = img.shape[1]

        src = np.float32([[500, 600], [1400, 600], [20, 1000], [1900, 1000]])
        dst = np.float32([[0, 0], [IMAGE_W, 0], [0, IMAGE_H], [IMAGE_W, IMAGE_H]])

        M = cv2.getPerspectiveTransform(src, dst) 
        Minv = cv2.getPerspectiveTransform(dst, src) 

        warped_img = cv2.warpPerspective(img,M,(IMAGE_W,IMAGE_H))
        cv2.imshow("Bird's eye view", warped_img)
        return warped_img, M, Minv
    
    def Trans_Pix2Camera(self, midpoint_x, midpoint_y):
        # Intrinsic matrix
        ## TODO ##
        #fing K_inv and Z
        fx = 0.0
        cx = 0.0
        fy = 0.0
        cy = 0.0
        Z = 0.0
        K_inv = np.linalg.inv(np.array([[fx, 0, cx],
                                        [0, fy, cy],
                                        [0, 0, 1]]))

        # Pixel coordinates in homogeneous form
        pixel_coords_homogeneous = np.array([midpoint_x, midpoint_y, 1])

        # Camera local coordinates
        camera_coords = Z * np.dot(K_inv, pixel_coords_homogeneous)

        return camera_coords

    def cleanup(self):
        print ("Shutting down vision node.")
        cv2.destroyAllWindows()
    

def main(args):
    try:
        ImageConverter()
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down vision node.")
        cv2.destryAllWindows()

if __name__ == '__main__':
    main(sys.argv)
