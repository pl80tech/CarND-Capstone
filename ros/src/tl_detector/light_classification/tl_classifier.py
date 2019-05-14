from styx_msgs.msg import TrafficLight
import traffic_light_detection as tl_detection
import rospy
import time

class TLClassifier(object):
    def __init__(self):
        #TODO load classifier
        self.path_to_graph = 'model/final/frozen_inference_graph.pb'
        self.path_to_label = 'model/final/tl_detect_label_map.pbtxt'
        self.detection_graph = None
        self.category_index = None
        self.detection_graph, self.category_index = tl_detection.get_final_model_info(self.path_to_graph, self.path_to_label)
        rospy.loginfo("Completed loading the detection graph and category index from specified model")

        # Create session for running tensorflow model
        self.sess = None
        self.sess = tl_detection.get_sess_for_inference(self.detection_graph)

        # Initial value
        self.detected_light = TrafficLight.UNKNOWN

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        #TODO implement light color prediction
        start_time = time.time()
        scores, classes = tl_detection.get_inference_of_image(image, self.detection_graph, self.sess)
        end_time = time.time()
        rospy.loginfo("Processing time for image inference is {} (s)".format(end_time - start_time))

        detected_class = self.category_index[classes[0]]['name']
        rospy.loginfo("Detected traffic light is {} with highest score = {}".format(detected_class, scores[0]))

        if detected_class == 'RED':
            self.detected_light = TrafficLight.RED
        elif detected_class == 'YELLOW':
            self.detected_light = TrafficLight.YELLOW
        elif detected_class == 'GREEN':
            self.detected_light = TrafficLight.GREEN
        else:
            self.detected_light = TrafficLight.UNKNOWN

        return self.detected_light
