#! /usr/bin/env python
# TODO: When implementing on F1-10, change it to "#! /usr/bin/env python3"

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
import os
import shutil
#----------------------------------------------------------------------
# Super parameter
# Path to the scripts
souce_path = "/home/stanley/f1_10_ws/f1tenth-main/src/f1tenth-sim/scripts/"
# Activate the function
object_detection = True
lane_detection = True
# Choose transform mode of coordinate. 
# "perspective_transform": 1  ; "trans_Pix2Camera": 2
transform_mode = 1
#----------------------------------------------------------------------
sys.path.append(souce_path)
sys.path.append(souce_path + "Detector/")
params_file = souce_path + 'depth_params.txt'
from yolo_detect_image import yolo_detect_image
from depth_function import get_depth_for_pixel
from depth_function import trans_Pix2Camera_v2
def renew_output_folder(folder_path):
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)
    os.makedirs(folder_path)

midpoint_path = souce_path + "output/midpoint/"
renew_output_folder(midpoint_path)
object_path = souce_path + "output/object_frame/"
renew_output_folder(object_path)
rgb_path = souce_path + "output/rgb_frame/"
renew_output_folder(rgb_path)

class ImageConverter:
    def __init__(self):
        self.node_name = "lane_detector"
        rospy.init_node(self.node_name)
        rospy.on_shutdown(self.cleanup)
        self.bridge = CvBridge()
        self.last_x = 0
        self.last_y = 0
        self.count = 0

        # Subscriber
        self.subcriber_rgb = message_filters.Subscriber('/D435I/color/image_raw', Image)
        sync = message_filters.ApproximateTimeSynchronizer([self.subcriber_rgb], 10, 1)
        sync.registerCallback(self.multi_callback)
        # Publisher
        self.midpoint_pose_pub = rospy.Publisher("/midpoint_pose", Point, queue_size=1)
        self.midpoint_img_pub = rospy.Publisher("/midpoint_img", Image, queue_size=1)
        self.object_detector_pub = rospy.Publisher("/object_detector", Image, queue_size=1)

    def multi_callback(self, rgb):
        try:
            rgb_frame = self.bridge.imgmsg_to_cv2(rgb, "bgr8")
        except CvBridgeError as e:
            rospy.logerr("CvBridge Error: {0}".format(e))
        point_msg = Point()
        # ----------------------------------------------------------------------
        # Main
        self.count += 1
        object_frame = rgb_frame
        detected = 0
        if self.count % 5 == 0 and object_detection: #if object_detection == True:
            object_list, object_frame, name, confidence_list = yolo_detect_image(rgb_frame, souce_path)
            print(object_list)
            if object_list == []:
                detected = 0
            else:
                detected = 1
                print(name)
                print(confidence_list)
                print("count",self.count)
                for detection in object_list:
                # Unpack the detection. Each detection is [left, top, right, bottom, classId, confidence]
                    left, top, right, bottom, classId, confidence = detection
                    midpoint_x = (left + right) // 2
                    midpoint_y = bottom
                    depth = get_depth_for_pixel(midpoint_x, midpoint_y, params_file)
                    if depth==0:
                        pass
                    print(f"Detected object: Class ID: {classId}, Confidence: {confidence:.2f}, Bottom Edge Midpoint: ({midpoint_x}, {midpoint_y})")
                    # X, Y, Z = trans_Pix2Camera_v2(midpoint_x, midpoint_y, depth, params_file)
                    # X, Y, Z = X/1000, Y/1000, Z/1000
                    # print(f"Midpoint depth ({midpoint_x}, {midpoint_y},Z, depth): {X}, {Y}, {Z}")
        if lane_detection == True:
            midpoint, midpoint_img = self.simple_lane_detector(rgb_frame)

            if transform_mode == 1:
                warped_img, M, Minv = self.perspective_transform(rgb_frame)
                point = np.array([midpoint[0], midpoint[1], 1])
                X, Y, Z = np.dot(M, point)
                Y = -Y + 0.3
                # print(f"perspective_tran ({midpoint[0]}, {midpoint[1]}): {X}, {Y}")
            elif transform_mode == 2:
                depth = get_depth_for_pixel(midpoint[0], midpoint[1], params_file)
                X, Y, Z = trans_Pix2Camera_v2(midpoint[0], midpoint[1], depth/1000, params_file)
                # X, Y, Z = self.trans_Pix2Camera(midpoint[0], midpoint[1], depth/1000)
                # print(f"trans_Pix2Camera ({midpoint[0]}, {midpoint[1]}): {X}, {Y}")
        # ----------------------------------------------------------------------
        # Publisher
        X = X +0.07
        # if X < 0.03 and X >-0.03:
        #     X = 0.0
        Y = Y + 0.5
        point_msg = Point()
        point_msg.x = X 
        point_msg.y = Y
        point_msg.z = detected
        self.midpoint_pose_pub.publish(point_msg)

        print(f"Depth at pixel ({midpoint[0]}, {midpoint[1]}): {X}, {Y}, {detected}")

        # TODO: Comment following when running on F1-10 to save computational resource
        # Output images result
        if self.count % 5 == 0:
            output_file_path = midpoint_path + str(self.count) + ".jpg"
            cv2.imwrite(output_file_path, midpoint_img)
            output_file_path = rgb_path + str(self.count) + ".jpg"
            cv2.imwrite(output_file_path, rgb_frame)
            output_file_path = midpoint_path +  "1.jpg"
            cv2.imwrite(output_file_path, midpoint_img)
            output_file_path = midpoint_path +  "2.jpg"
            cv2.imwrite(output_file_path, rgb_frame)
            output_file_path = object_path + str(self.count) + ".jpg"
            cv2.imwrite(output_file_path, object_frame)
            
        # Publish the detected object
        object_img = CvBridge().cv2_to_imgmsg(object_frame, "bgr8")
        self.object_detector_pub.publish(object_img)
        # Publish the image
        ros_img = CvBridge().cv2_to_imgmsg(midpoint_img, "bgr8")
        self.midpoint_img_pub.publish(ros_img)


    def simple_lane_detector(self, frame):
        mask_list = self.create_mask(frame)
        # Thresh the lane through only color
        lane = self.thresh_lane(frame)
        # Select the region of interest
        segmented_image, mask = self.region_of_interest(lane, mask_list)
        # Find the midpoint
        midpoint_list, midpoint_img = self.find_lane_midpoint(segmented_image)

        try:
            if len(midpoint_list) > 2:
                m = 2
                n = 1
                midpoint = ((m * np.array(midpoint_list[-3]) + n * np.array(midpoint_list[-2])) / (n + m)).astype(int)
            elif len(midpoint_list) > 1:
                m = 2
                n = 1
                midpoint = ((m * np.array(midpoint_list[-2]) + n * np.array(midpoint_list[-1])) / (n + m)).astype(int)
            else:
                midpoint = midpoint_list[-1]
            self.last_x = midpoint[0]
            self.last_y = midpoint[1]
        except:
            print("No detected waypoint!")
            midpoint = [self.last_x, self.last_y]

        # Optional: Draw the midpoints on the mask for visualization
        cv2.circle(midpoint_img, tuple(midpoint), 5, (0, 255, 0), -1)

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

        max_density = 0
        max_density_index = 0
        left_bound = None
        right_bound = None


        # Iterate over each horizontal slice
        for i in range(num_slices):
            # Calculate the y-coordinate of this slice
            y_coord = i * increment

            # Find all non-zero x-coordinates at this y-coordinate
            white_pixels = np.where(lane_mask[y_coord, :] > 0)[0]

            #Find the place where the white area has the highest density 
            #Then find the highest density area as the midpoint's boundry can calculate the midpoint

            if len(white_pixels) > max_density:
                max_density = len(white_pixels)
                max_density_index = i
                left_bound = white_pixels[0] if len(white_pixels) > 0 else None   
                right_bound = white_pixels[-1] if len(white_pixels) > 0 else None
                midpoint = ((left_bound+ right_bound) // 2, y_coord)

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

        combined_mask = cv2.bitwise_or(ColorOutput1, ColorOutput2)
        combined_mask = cv2.bitwise_or(combined_mask, ColorOutput3)

        # Find the indices of the white pixels
        white_pixels_indices = np.column_stack(np.where(combined_mask == 255))

        # Calculate the covariance matrix of the white pixel positions
        covariance_matrix = np.cov(white_pixels_indices, rowvar=False)
        covariance_value = np.mean(covariance_matrix)
        # print("covariance_value:", covariance_value)

        if covariance_value > 7000:
            ratio = 0.9
        else:
            ratio = 0.2

        # Define the size of each piece (block size)
        piece_width = combined_mask.shape[1] // 30  # for example, divide the image into 10 pieces horizontally
        piece_height = combined_mask.shape[0] // 30  # divide the image into 10 pieces vertically

        # Define the threshold for minimum white pixels in a piece
        white_pixel_threshold = (piece_width * piece_height) * ratio  # for example, 10% of the piece area

        # Iterate over the image piece by piece
        for y in range(0, combined_mask.shape[0], piece_height):
            for x in range(0, combined_mask.shape[1], piece_width):
                # Extract the piece
                piece = combined_mask[y:y+piece_height, x:x+piece_width]
                
                # Count the white pixels in the piece
                white_pixels = cv2.countNonZero(piece)
                
                # If white pixels are less than the threshold, change all pixels to black
                if white_pixels < white_pixel_threshold:
                    combined_mask[y:y+piece_height, x:x+piece_width] = 0

        # combined_image = cv2.hconcat([ColorOutput1, ColorOutput2, combined_mask,combined_mask1])
        # cv2.imshow("combined_image", combined_image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        return combined_mask
    
    def read_params(self, file_path):
        """
        Read calibration and depth parameters from a file.

        Parameters:
        file_path (str): Path to the file containing the parameters.

        Returns:
        dict: Dictionary containing the parameters.
        """
        params = {}
        with open(file_path, 'r') as f:
            for line in f:
                key, value = line.strip().split(': ')
                if key in ['Image Height', 'Image Width']:
                    params[key] = int(value)
                elif key in ['Camera Height']:
                    params[key] = float(value)
                elif key in ['Camera Matrix', 'Distortion Coefficients', 'Ground Plane Normal']:
                    params[key] = np.array(eval(value))
        return params

    def calculate_depth_at_pixel(self, x, y, height, width, normal, camera_height, camera_matrix):
        if y < height // 2 or y >= height or x < 0 or x >= width:
            return "Pixel out of range for depth calculation"
        ray = np.linalg.inv(camera_matrix) @ np.array([x, y, 1])
        cos_theta = np.dot(ray, normal) / (np.linalg.norm(ray) * np.linalg.norm(normal))
        depth = camera_height / cos_theta
        return depth
    
    def read_depth(self, x_pixel, y_pixel):
        # Path to the file containing the parameters
        # depth_file = 'calib_data/depth_params.txt'
        depth_file_path = souce_path + "Detector/depth_params.txt"

        # Read parameters
        params = self.read_params(depth_file_path)

        # Calculate depth
        depth = self.calculate_depth_at_pixel(x_pixel, y_pixel, params['Image Height'], params['Image Width'], params['Ground Plane Normal'], params['Camera Height'], params['Camera Matrix'])
        return depth

    def perspective_transform(self, img, verbose=False):
        
        img = img.astype('uint8') * 255
        
        IMAGE_H = img.shape[0]
        IMAGE_W = img.shape[1]

        # src = np.float32([[208-int(IMAGE_W/2), IMAGE_H-292], [412-int(IMAGE_W/2), IMAGE_H-285], [178-int(IMAGE_W/2), IMAGE_H-338], [550-int(IMAGE_W/2), IMAGE_H-322]])
        # dst = np.float32([[0.292, 2.197], [0.470, 2.197], [0.203, 1.18], [0.559, 1.18]])


        src = np.float32([[208, 292], [412 , 285], [178 , 338], [550 , 322]])
        dst = np.float32([[-0.292, 2.197], [0.470, 2.197], [-0.203, 1.18], [0.559, 1.18]])

        M = cv2.getPerspectiveTransform(src, dst) 
        Minv = cv2.getPerspectiveTransform(dst, src) 

        warped_img = cv2.warpPerspective(img,M,(IMAGE_W,IMAGE_H))
        # print("warped_img:", warped_img.shape)
        # cv2.imshow("Bird's eye view", warped_img)
        return warped_img, M, Minv

    def trans_Pix2Camera(self, midpoint_x, midpoint_y, depth):
        # Intrinsic matrix
        #fing K_inv and depth
        fx = 605.04541015625
        cx = 322.3511047363281
        fy = 604.6482543945312
        cy = 247.38180541992188
        # The distance between camera and center of F1-10
        offset_y = 0.3
        # try:
        #     depth = self.read_depth(midpoint_x, midpoint_y)
        # except:
        #     depth = 1
        #     print("No estimated depth!")

        K_inv = np.linalg.inv(np.array([[fx, 0, cx],
                                        [0, fy, cy],
                                        [0, 0, 1]]))

        # Pixel coordinates in homogeneous form
        pixel_coords_homogeneous = np.array([midpoint_x, midpoint_y, 1])

        # Camera local coordinates
        camera_coords = depth * np.dot(K_inv, pixel_coords_homogeneous)

        return camera_coords[0], camera_coords[1], camera_coords[2]

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



# Append

 # def find_lane_midpoint(self, lane_mask):
    #     # List to hold midpoints
    #     midpoints = []
    #     confidences = []

    #     # Get the height of the mask
    #     height = lane_mask.shape[0]

    #     # Define the number of horizontal slices
    #     num_slices = 10  # or however many slices you want to divide your image into

    #     # Calculate the increment for each horizontal slice
    #     increment = height // num_slices

    #     # Iterate over each horizontal slice
    #     for i in range(num_slices):
    #         # Calculate the y-coordinate of this slice
    #         y_coord = i * increment

    #         # Find all non-zero x-coordinates at this y-coordinate
    #         white_pixels = np.where(lane_mask[y_coord, :] > 0)[0]

    #         if len(white_pixels) > 0:
    #             # Calculate the midpoint for this slice
    #             leftmost = white_pixels[0]
    #             rightmost = white_pixels[-1]
    #             midpoint = ((leftmost + rightmost) // 2, y_coord)

    #             # Add the midpoint to the list of midpoints
    #             midpoints.append(midpoint)
    #             confidences.append(len(white_pixels))

    #     # # Pair each midpoint with its corresponding confidence
    #     # midpoint_confidence_pairs = list(zip(midpoints, confidences))

    #     # # Sort the pairs based on confidence, with the highest confidence first
    #     # midpoint_confidence_pairs.sort(key=lambda x: x[1], reverse=True)

    #     # # Assuming you want the single midpoint with the highest confidence
    #     # highest_confidence_midpoint = midpoint_confidence_pairs[0][0]

    #     # Optional: Draw the midpoints on the mask for visualization
    #     midpoint_visualized = cv2.cvtColor(lane_mask, cv2.COLOR_GRAY2BGR)

    #     for point in midpoints:
    #         cv2.circle(midpoint_visualized, point, 5, (0, 0, 255), -1)

    #     return midpoints, midpoint_visualized
