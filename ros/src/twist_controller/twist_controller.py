import rospy
from yaw_controller import YawController
from pid import PID
from lowpass import LowPassFilter

GAS_DENSITY = 2.858
ONE_MPH = 0.44704

class Controller(object):
    def __init__(self, vehicle_mass, fuel_capacity , brake_deadband, decel_limit, accel_limit, 
                 wheel_radius, wheel_base, steer_ratio, max_lat_accel,max_steer_angle):
        # Initialze yaw controller
        min_speed = 0.1
        self.yaw_controller = YawController(wheel_base, steer_ratio, min_speed, max_lat_accel, max_steer_angle)
        
        # Throttle's PID controller parameters
        kp = 0.3
        ki = 0.1
        kd = 0.
        
        # Maximum and minimum values for Throttle
        min_throttle = 0.
        max_throttle = 0.2
        
        # Initialze PID controller for throttle
        self.throttle_controller = PID(kp, ki, kd, min_throttle, max_throttle)
        
        # Lowpass filter parameters for high frequency noise
        tau = 0.5                                   # 1/(2pi*tau) = cutoff frequency
        ts = 0.02                                   # sample time
        
        # Initialze low pass filter to be used for velocity filtering
        self.vel_lpf = LowPassFilter(tau, ts)
        
        self.vehicle_mass = vehicle_mass
        self.fuel_capacity = fuel_capacity
        self.brake_deadband = brake_deadband
        self.decel_limit = decel_limit
        self.accel_limit = accel_limit
        self.wheel_radius = wheel_radius
        
        self.last_time = rospy.get_time()
        
    def control(self, current_vel, dbw_enabled, linear_vel, angular_vel):
        """ Calculates value of throttle, brake and steering angle 
            based on current velocity and constant parameters
        Args:
            current_vel : current velocity of ego car
            dbw_enabled : dbw enabled or not
            linear_vel  : linear velocity of ego car
            angular_vel : angular velocity of ego car
        Returns:
            int: throttle, brake, steer
        """

        # if dbw_enabled is not True, the car will NOT be controlled by the controller.
        if not dbw_enabled:
            self.throttle_controller.reset()
            return 0., 0., 0.
        
        # Filter current velocity with the Low Pass Filter
        current_vel = self.vel_lpf.filt(current_vel)
        
        # Get a steering value through the yaw controller
        steering = self.yaw_controller.get_steering(linear_vel, angular_vel, current_vel)

        vel_error = linear_vel - current_vel
        self.last_vel = current_vel
        
        current_time = rospy.get_time()
        sample_time = current_time - self.last_time
        self.last_time = current_time
        
        # Get a throttle value through the PID
        throttle = self.throttle_controller.step(vel_error, sample_time)
        brake = 0
        
        # If the "goal" velocity is 0 (maybe a red traffic light), I will send a brake transition
        # If velocity nearly to 0, apply the MAX brake force to make the car totally stops
        if linear_vel == 0. and current_vel < 0.1:
            throttle = 0
            brake = 700                                                 # N*m this is the required torque necessaty for Carla to totally brake.
        # If car is going to fast, slowly decrease the car velocity    
        elif throttle < .1 and vel_error < 0:
            throttle = 0
            decel = max(vel_error, self.decel_limit)                    # make the car not Hard Brake, too uncomfortable
            brake = abs(decel) * self.vehicle_mass*self.wheel_radius    # Torque N*m
            
        return throttle, brake, steering