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
        output_dict = tl_detection.get_detected_objects(image, self.detection_graph, self.category_index)
        end_time = time.time()
        rospy.loginfo("Processing time for image inference is {} (s)".format(end_time - start_time))

        classes = output_dict['detection_classes']
        scores = output_dict['detection_scores']
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
