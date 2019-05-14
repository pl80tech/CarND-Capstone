#!/usr/bin/env python
import rospy
from std_msgs.msg import Int32
from geometry_msgs.msg import PoseStamped, Pose
from styx_msgs.msg import TrafficLightArray, TrafficLight
from styx_msgs.msg import Lane
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from light_classification.tl_classifier import TLClassifier
import tf
import cv2
import yaml
import sys
from scipy.spatial import KDTree
from timeit import default_timer as timer
import os
import traffic_light_detection as tl_detection

STATE_COUNT_THRESHOLD = 3

class TLDetector(object):
    def __init__(self):
        rospy.init_node('tl_detector')

        self.pose = None
        self.waypoints = None
        self.camera_image = None
        self.lights = []
        self.waypoints_2d = None
        self.waypoint_tree = None

        sub1 = rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        sub2 = rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)

        '''
        /vehicle/traffic_lights provides you with the location of the traffic light in 3D map space and
        helps you acquire an accurate ground truth data source for the traffic light
        classifier by sending the current color state of all traffic lights in the
        simulator. When testing on the vehicle, the color state will not be available. You'll need to
        rely on the position of the light and the camera image to predict it.
        '''
        sub3 = rospy.Subscriber('/vehicle/traffic_lights', TrafficLightArray, self.traffic_cb)
        sub6 = rospy.Subscriber('/image_color', Image, self.image_cb)

        config_string = rospy.get_param("/traffic_light_config")
        self.config = yaml.load(config_string)

        self.upcoming_red_light_pub = rospy.Publisher('/traffic_waypoint', Int32, queue_size=1)

        self.bridge = CvBridge()
        self.light_classifier = TLClassifier()
        self.listener = tf.TransformListener()

        self.state = TrafficLight.UNKNOWN
        self.last_state = TrafficLight.UNKNOWN
        self.last_wp = -1
        self.state_count = 0

        # Check whether to use light state from simulator (argument from command line )
        self.use_simulator_light_state = sys.argv[1] == 'true'

        # Check whether to save camera image for training and testing the model
        self.save_camera_image = sys.argv[2] == 'true'

        # Check whether to save inference image with detected objects
        self.save_inference_image = sys.argv[3] == 'true'

        # Save camera image and light state to csv file
        self.csvfile = open("lightstate.csv", 'w')

        # Get detection graph and category index of model#3 (faster_rcnn_resnet101_coco_2018_01_28)
        if (self.save_inference_image):
            model = 3
            self.detection_graph = None
            self.category_index = None
            self.detection_graph, self.category_index = tl_detection.get_model_info(model)

        # Counter to skip processing camera image
        self.counter = 0

        rospy.spin()

    def pose_cb(self, msg):
        self.pose = msg

    def waypoints_cb(self, waypoints):
        self.waypoints = waypoints
        if not self.waypoints_2d:
            self.waypoints_2d = [[waypoint.pose.pose.position.x, waypoint.pose.pose.position.y] for waypoint in waypoints.waypoints]
            self.waypoint_tree = KDTree(self.waypoints_2d)

    def traffic_cb(self, msg):
        self.lights = msg.lights

    def image_cb(self, msg):
        """Identifies red lights in the incoming camera image and publishes the index
            of the waypoint closest to the red light's stop line to /traffic_waypoint

        Args:
            msg (Image): image from car-mounted camera

        """
        self.has_image = True
        self.camera_image = msg
        light_wp, state = self.process_traffic_lights()

        '''
        Publish upcoming red lights at camera frequency.
        Each predicted state has to occur `STATE_COUNT_THRESHOLD` number
        of times till we start using it. Otherwise the previous stable state is
        used.
        '''
        if self.state != state:
            self.state_count = 0
            self.state = state
        elif self.state_count >= STATE_COUNT_THRESHOLD:
            self.last_state = self.state
            light_wp = light_wp if state == TrafficLight.RED else -1
            self.last_wp = light_wp
            self.upcoming_red_light_pub.publish(Int32(light_wp))
        else:
            self.upcoming_red_light_pub.publish(Int32(self.last_wp))
        self.state_count += 1

    def get_closest_waypoint(self, x, y):
        """Identifies the closest path waypoint to the given position
            https://en.wikipedia.org/wiki/Closest_pair_of_points_problem
        Args:
            pose (Pose): position to match a waypoint to

        Returns:
            int: index of the closest waypoint in self.waypoints

        """
        #TODO implement
        closest_idx = self.waypoint_tree.query([x, y], 1)[1]
        return closest_idx

    def get_light_state(self, light):
        """Determines the current color of the traffic light

        Args:
            light (TrafficLight): light to classify

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        if (self.use_simulator_light_state):
            # For testing, just return the light state
            return light.state
        else:
            if(not self.has_image):
                self.prev_light_loc = None
                return self.last_state

            rospy.loginfo("--------------------------------------------")

            # Skip processing the classification
            self.counter += 1
            if self.counter % 2 == 0:
                rospy.loginfo("counter = {} --> skip processing the classification".format(self.counter))
                return self.last_state

            cv_image = self.bridge.imgmsg_to_cv2(self.camera_image, "bgr8")

            # Save camera image if specified
            if (self.save_camera_image):
                time_info = timer()
                filename = os.path.join("./dataset/", "camera_image_" + "%s.jpg" % time_info)
                #rospy.loginfo("filename = {}".format(filename))
                cv2.imwrite(filename, cv_image)
                row = "{},{}\n".format(filename, light.state)
                self.csvfile.write(row)
                rospy.loginfo("light.state = {}".format(light.state))

                # Detect and save inference images with model#3 (faster_rcnn_resnet101_coco_2018_01_28)
                if (self.save_inference_image):
                    image_path = filename
                    inf_image_path = os.path.join("./dataset_inference/", "camera_image_inf_" + "%s.jpg" % time_info)
                    tl_detection.detect_and_save_image_model(image_path, inf_image_path, self.detection_graph, self.category_index)

            #Get and return classification result
            detected_light = self.light_classifier.get_classification(cv_image)
            rospy.loginfo("Detected traffic light = {}".format(detected_light))

            return detected_light

    def process_traffic_lights(self):
        """Finds closest visible traffic light, if one exists, and determines its
            location and color

        Returns:
            int: index of waypoint closes to the upcoming stop line for a traffic light (-1 if none exists)
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        closest_light = None
        line_wp_idx = None

        # List of positions that correspond to the line to stop in front of for a given intersection
        stop_line_positions = self.config['stop_line_positions']
        if(self.pose):
            car_wp_idx = self.get_closest_waypoint(self.pose.pose.position.x, self.pose.pose.position.y)

            #TODO find the closest visible traffic light (if one exists)
            diff = len(self.waypoints.waypoints)
            for i, light in enumerate(self.lights):
                # Get stop line waypoint index
                line = stop_line_positions[i]
                temp_wp_idx = self.get_closest_waypoint(line[0], line[1])
                # Find closest stop line waypoint index
                d = temp_wp_idx - car_wp_idx
                if d >= 0 and d < diff:
                    diff = d
                    closest_light = light
                    line_wp_idx = temp_wp_idx

        if closest_light:
            state = self.get_light_state(closest_light)
            return line_wp_idx, state

        return -1, TrafficLight.UNKNOWN

if __name__ == '__main__':
    try:
        TLDetector()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start traffic node.')
