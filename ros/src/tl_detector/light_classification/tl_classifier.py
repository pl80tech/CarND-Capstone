from styx_msgs.msg import TrafficLight
import traffic_light_detection as tl_detection
import rospy
import time
import cv2
import yaml
import numpy as np
from keras import backend as K
from keras.models import load_model

class TLClassifier(object):
    def __init__(self):
        #TODO load classifier
        # Initial value
        self.detected_light = TrafficLight.UNKNOWN

        config_file = open("tl_classification_config.yaml")
        config = yaml.load(config_file)
        self.use_pretrained_tf_model = config['use_pretrained_tf_model']
        
        if (self.use_pretrained_tf_model):
            rospy.loginfo("Use pre-trained and customized model for traffic light classification")
            self.path_to_graph = config['path_to_graph']
            self.path_to_label = config['path_to_label']
            self.detection_threshold = config['detection_threshold']
            self.detection_graph = None
            self.category_index = None
            self.detection_graph, self.category_index = tl_detection.get_final_model_info(self.path_to_graph, self.path_to_label)
            rospy.loginfo("Completed loading the detection graph and category index from specified model")

            # Create session for running tensorflow model
            self.sess = None
            self.sess = tl_detection.get_sess_for_inference(self.detection_graph)

            # Run session to detect a test image
            # Just a temporary process to save time for image inference of actual camera images
            # since 1st inference takes more time than next inferences
            test_image = cv2.imread('data/test.jpg')
            start_time = time.time()
            test_scores, test_classes = tl_detection.get_inference_of_image(test_image, self.detection_graph, self.sess)
            end_time = time.time()
            rospy.loginfo("Processing time for 1st inference is {} (s)".format(end_time - start_time))
        else:
            # Use newly built model
            rospy.loginfo("Use newly built model for traffic light classification")
            self.path_to_model = config['path_to_model']
            self.model = load_model(self.path_to_model)
            self.model._make_predict_function()
            self.graph = K.tf.get_default_graph()

        # Close the config file
        config_file.close()

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        #TODO implement light color prediction
        if (self.use_pretrained_tf_model):
            start_time = time.time()
            scores, classes = tl_detection.get_inference_of_image(image, self.detection_graph, self.sess)
            end_time = time.time()
            rospy.loginfo("Processing time for image inference is {} (s)".format(end_time - start_time))

            # Only return classification result if inference score is better than threshold
            if (scores[0] > self.detection_threshold):
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
            else:
                rospy.loginfo("No traffic light is detected. Keep the last light state")
        else:
            # Use newly built model
            image = cv2.resize(image, (320, 240))
            image_reshape = np.reshape(image,  (1, 240, 320, 3))
            rospy.logdebug("Start to predict traffic light")
            try:
                with self.graph.as_default ():
                    # Get prediction result
                    predict_list = self.model.predict(image_reshape)
                    predict_result = np.argmax(predict_list)

                    # Set classification result
                    if (predict_result == 0):
                        self.detected_light = TrafficLight.RED
                    elif (predict_result == 1):
                        self.detected_light = TrafficLight.YELLOW
                    elif (predict_result == 2):
                        self.detected_light = TrafficLight.GREEN
                    else:
                        self.detected_light = TrafficLight.UNKNOWN

            except Exception as e:
                rospy.logerr(e)
                return TrafficLight.UNKNOWN

        return self.detected_light
