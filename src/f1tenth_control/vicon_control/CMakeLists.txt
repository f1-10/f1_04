cmake_minimum_required(VERSION 3.0.2)
project(vicon_control)

find_package(catkin REQUIRED COMPONENTS
  cv_bridge
  roscpp
  rospy
  sensor_msgs
  std_msgs
  tf
  geometry_msgs
  ackermann_msgs
  message_generation
)

catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES f1tenth
   CATKIN_DEPENDS
   geometry_msgs
   rospy
   sensor_msgs
   std_msgs
   tf
   message_runtime
   #  DEPENDS system_lib
)

include_directories(
# include
  ${catkin_INCLUDE_DIRS}
)

install(PROGRAMS
  ./scripts/lane_detector.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
