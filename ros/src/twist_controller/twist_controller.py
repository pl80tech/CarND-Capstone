
GAS_DENSITY = 2.858
ONE_MPH = 0.44704


class Controller(object):
    def __init__(self, vehicle_mass, fuel_capacity, brake_deadband, decel_limit,
    	accel_limit, wheel_radius, wheel_base, steer_ratio, max_lat_accel, max_steer_angle):

        # TODO: Implement
		self.yaw_controller = YawController(wheel_base, steer_ratio, 0.1, max_lat_accel, max_steer_angle)

		kp = 0.3
		ki = 0.1
		kd = 0.
		mn = 0. # Minimum throttle value
		mx = 0.2 # Maximum throttle value
		self.throttle_controller = PID(kp, ki, kd, mn, mx)

		tau = 0.5 # 1/(2pi*tau) = cutoff frequency
		ts = 0.2 # Sample time
		self.vel_lpf = LowPassFilter(tau, ts)

		self.vehicle_mass = vehicle_mass
		self.fuel_capacity = fuel_capacity
		self.brake_deadband = brake_deadband
		self.decel_limit = decel_limit
		self.accel_limit = accel_limit
		self.wheel_radius = wheel_radius

		self.last_time = rospy.get_time()

    def control(self, *args, **kwargs):
        # TODO: Change the arg, kwarg list to suit your needs
        # Return throttle, brake, steer
        return 1., 0., 0.
