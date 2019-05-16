'''
Input
- Test image
- Model to be tested

Output
- Processing time for image inference
- Detected classes with scores
'''
import traffic_light_detection as tl_detection
import time
import sys
import cv2
import os
import tensorflow as tf

# Main function
def main():
	# The model to be tested 
    model = int(sys.argv[1])

    # Path to specified test image
    imagepath = sys.argv[2]

    # Get model info
    if (model == 0):
        # Retrained/final model
        path_to_graph = 'model/final/frozen_inference_graph.pb'
        path_to_label = 'model/final/tl_detect_label_map.pbtxt'
        detection_graph, category_index = tl_detection.get_final_model_info(path_to_graph, path_to_label)
    else:
        # Existed/Pre-trained model
        detection_graph, category_index = tl_detection.get_model_info(model)

    # Get session for classification
    sess = tl_detection.get_sess_for_inference(detection_graph)

    # Get image list from specified path
    images = os.listdir(imagepath)

    # Load and process each image in the list
    for fname in images:
        print("--------------------")
        print("processing image: " + fname)

        # Load image
        image = cv2.imread(imagepath + fname)

        # Get inference result
        start_time = time.time()
        scores, classes = tl_detection.get_inference_of_image(image, detection_graph, sess)
        end_time = time.time()

        # Output processing time for inference
        print("processing time = " + str(end_time - start_time))

        # Output inference result
        detected_class = category_index[classes[0]]['name']
        print("Object " + detected_class + " is detected with highest score " + str(scores[0]))

if __name__ == "__main__":
    print(tf.__version__)
    main()
