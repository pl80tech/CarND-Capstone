import os
import sys
import numpy as np
import cv2
import tensorflow as tf
from keras import backend as K
from keras.models import load_model

# Run prediction for all images in specified folder and output images with classification result
def runPrediction(model, input_path, output_path):
	images = os.listdir(input_path)
	images.sort()
	for image in images:
		imagepath = input_path + '/' + image
		img = cv2.imread(imagepath)
		img_resize = cv2.resize(img, (320, 240))
		img_reshape = np.reshape(img_resize, (1, 240, 320, 3))
		score_list = model.predict(img_reshape)
		light_type = np.argmax(score_list)
		#print("light_type = " + str(light_type))

		# Put text to image
		if light_type == 0:
			text = 'Traffic light: RED'
			color = (0,0,255)
		elif light_type == 1:
			text = 'Traffic light: YELLOW'
			color = (0,255,255)
		elif light_type == 2:
			text = 'Traffic light: GREEN'
			color = (0,255,0)
		else:
			text = 'Traffic light: UNKNOWN'
			color = (255,0,0)
		font = cv2.FONT_HERSHEY_SIMPLEX
		cv2.putText(img, text, (50, 50), font, 2, color, thickness=2)

		# Save image
		cv2.imwrite(output_path + '/' + image, img)

# Main function
def main():
	# Get parameters from command lines
	model_file = sys.argv[1]
	dataset_path = sys.argv[2]
	outFolder = sys.argv[3]

	# Create output folder
	if not os.path.exists(outFolder):
		os.makedirs(outFolder)

	# Load model
	model = load_model(model_file)
	model._make_predict_function()
	graph = K.tf.get_default_graph()

	try:
		with graph.as_default ():
			runPrediction(model, dataset_path, outFolder)
	except Exception as e:
		print("Exception")

if __name__ == "__main__":
    main()