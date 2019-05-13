from styx_msgs.msg import TrafficLight
import traffic_light_detection as tl_detection

class TLClassifier(object):
    def __init__(self):
        #TODO load classifier
        self.path_to_graph = 'model/final/frozen_inference_graph.pb'
        self.path_to_label = 'model/final/tl_detect_label_map.pbtxt'
        self.detection_graph = None
        self.category_index = None
        self.detection_graph, self.category_index = tl_detection.get_final_model_info(self.path_to_graph, self.path_to_label)

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        #TODO implement light color prediction
        output_dict = tl_detection.get_detected_objects(image, self.detection_graph, self.category_index)
        classes = output_dict['detection_classes']
        scores = output_dict['detection_scores']
        detected_class = self.category_index[classes[0]]['name']

        return TrafficLight.UNKNOWN
