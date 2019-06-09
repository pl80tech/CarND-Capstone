import os
import sys
import tensorflow as tf
from keras import backend as K
from keras.models import load_model

# Run prediction for all images in specified folder and output images with classification result
def runPrediction(model, input_path, output_path):

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