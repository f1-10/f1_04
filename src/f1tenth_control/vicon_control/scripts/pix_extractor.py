#!/usr/bin/env python
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge,CvBridgeError
import cv2

#!/usr/bin/env python
import rospy
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image

# Callback function for mouse clicks
def click_event(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        print("Coordinates of pixel: X: ", x, " Y: ", y)
        font = cv2.FONT_HERSHEY_PLAIN
        cv2.putText(cv_image, str(x) + ',' + str(y), (x, y), font, 1, (255, 0, 0), 2)
        cv2.imshow('Image', cv_image)

# Callback function for ROS image message
def image_callback(msg):
    global cv_image
    try:
        cv_image = bridge.imgmsg_to_cv2(msg, "bgr8")
    except cv_bridge.CvBridgeError as e:
        print(e)

    cv2.imshow('Image', cv_image)
    cv2.setMouseCallback('Image', click_event)
    cv2.waitKey(1)

# Initialize ROS node
rospy.init_node('image_coordinate_extractor', anonymous=True)
bridge = CvBridge()

# Subscribe to the image topic
image_topic = "/D435I/color/image_raw"  # change to your topic
rospy.Subscriber(image_topic, Image, image_callback)

# Keep the script alive
try:
    rospy.spin()
except KeyboardInterrupt:
    print("Shutting down")
cv2.destroyAllWindows()

