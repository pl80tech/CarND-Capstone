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

# Main function
def main():
	# The model to be tested 
    model = int(sys.argv[1])

    # Path to specified test image
    imagepath = sys.argv[2]

    # Get model info
    detection_graph, category_index = tl_detection.get_model_info(model)

    # Load image
    image = cv2.imread(imagepath)

    # Get inference result
    start_time = time.time()
    output_dict = tl_detection.get_detected_objects(image, detection_graph, category_index)
    end_time = time.time()

    # Output processing time for inference
    print("processing time = " + str(end_time - start_time))

    # Output inference result
    classes = output_dict['detection_classes']
    scores = output_dict['detection_scores']
    detected_class = category_index[classes[0]]['name']
    print("Object " + detected_class + " is detected with highest score " + str(scores[0]))

if __name__ == "__main__":
    main()
