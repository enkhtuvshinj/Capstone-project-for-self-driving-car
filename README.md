## Self Driving Car Engineer ND - Capstone Project


### Project Team
| Team position        | Name           | Email  |
| ------------- | ------------- | -----:|
| Team leader      | Igor Quint | igorquint@gmail.com |
| Team member-1      | Felipe Rojas      |   epilefrojasleon@gmail.com |
| Team member-2 | Enkhtuvshin Janchivnyambuu      |    enkhtuvshinj@gmail.com |
| Team member-3 | Sushrutha Krishnamurthy      |    sushukrish@gmail.com |


### Project Overview
The project was executed by the above team. The code development was shared within the team and tested independently by each team member. The results were discussed, and bug fixed in stages.

The code development is based on the instructions provided by the walkthrough videos. The coding was implemented in 4-steps:

* **Step-1 Partial Waypoint**-
Partial implementation of the waypoint_updater.py with the goal of generating waypoints, to have the vehicle follow the waypoints, ignoring the traffic lights.

* **Step-2 DBW**-
Implementing the twist_controller such that the twist commands are converted into accelerator, brake and steering command to control the vehicle.

* **Step-3 Traffic light detection**-
Initially the true traffic state was used to implement vehicle deceleration at red traffic light. Later traffic light detection and traffic light identification was implemented through tensor flow classification

* **Step-4 Waypoint updater**-
The waypoint updater logic now has the topic traffic_waypoint as subscriber and fully implements this information into the waypoints calculation

Below are details of the code-implementation in the different modules
  

### Waypoint Updater Node
This node publishes final waypoints which contain the right linear and angular target velocities. Waypoint updater uses base waypoints and the current position, as well as the detected traffic lights (through the traffic_waypoint topic subscription).

Base waypoints are provided by a static csv file. The current position is obtained through a message by the simulator. The final waypoints are published and picked up by the waypoint follower node.

The function set_waypoint velocity adjust the linear velocity of a waypoint if for example a red traffic light is nearby. The distance function is used to calculate the distance between waypoints which is used to properly decelerate when a red traffic light is close.


### Twist Controller
The libraries PID, yaw_controller and lowpass are imported at the start. The class controller is implemented and the arguments of the class are set.

For the steering control the yaw controller is called and provided with the values wheel_base, steer_ratio, min_speed, max_lat_accel, max_steer_angle. The minimum speed for steering calculation has been set to 0.1

For the throttle control the PID controller is called. The Kp, Ki and Kd values as well as the the max- and min- throttle values are set after a few experimentations. 

The current velocity is filtered through a low pass filter. The cutoff frequency for the filtering is also set

The function control is defined to return the throttle, brake and steering values, based on the inputs: self, current_vel, dbw_enabled, linear_vel, angular_vel

If the simulator returns dbw_ enabled as inactive the return values are zero and the car will not be controlled by the controller

Otherwise the yaw controller and the PID controllers control the steering and the throttle respectively. 

If the target velocity is zero and the vehicle current velocity is below 0.1 brakes are applied with a constant torque of 700 Nm to prevent the car from rolling.  Else if, the car is above the threshold speed and vehicle deceleration is required, the brake torque is calculated based on the set deceleration limit, vehicle mass, wheel radius and the error between the current and the desired velocities.

The calculated throttle, brake and steer values are returned 

### DBW Node

### Traffic Light Detection
This node predicts a color of traffic light, that ego vehicle is approaching, using trained **SSD MobileNet v1** of [TensorFlow Object Detection API](https://github.com/tensorflow/models/tree/master/research/object_detection) for traffic light detection by feeding camera data. 
The traffic light detection package consists of two python files:

* **tl_detector.py** - 
This python file processes the incoming traffic light data and camera images. 
It uses the light classifier implemented in **tl_classifier.py** to get a upcoming traffic light state, and publishes the location of traffic light stop line if the traffic light state is RED.

* **tl_classifier.py** -
This file contains implementation of the SSD MobileNet v1 classifier. 
It receives the image from camera and returns the traffic light state: RED, YELLOW, GREEN and UNKNOWN.
For this project, we have used the training datasets and model config prepared by Udacity student [Vatsal Srivastava](https://github.com/coldKnight/TrafficLight_Detection-TensorFlowAPI) in SSD MobileNet v1 classifier.

For more detailed information on how to install and train TensorFlow Object Detection API, please refer to [instructions here](./instructions_tf_model.md).

The traffic light detection node subscribes to four topics:

* **/base_waypoints** provides the complete list of waypoints for the course.
* **/current_pose** can be used to determine the vehicle's location.
* **/image_color** which provides an image stream from the car's camera. These images are used to determine the color of upcoming traffic lights.
* **/vehicle/traffic_lights** provides the (x, y, z) coordinates of all traffic lights.

The node publishes the index of the waypoint for nearest upcoming red light's stop line to a single topic **/traffic_waypoint**.
