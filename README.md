# f1_04
The code implemented on the real vehicle

Initial setup:

download weights from :https://pjreddie.com/media/files/yolov3-tiny.weights

(place it in the Detector folder)


Step 1  launch the sensor
```bash
source devel/setup.bash
roslaunch racecar sensors.launch 
```

Step 2 launch the joystick controller
```bash
source devel/setup.bash
roslaunch racecar teleop.launch
```

Step 3 run the lane detection
```bash
source devel/setup.bash
rosrun vicon_control lane_detector.py
```

Step 4 run the pure pursuit controller
```bash
source devel/setup.bash
rosrun vicon_control pure_pursuit_controller.py
```
