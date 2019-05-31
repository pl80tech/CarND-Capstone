# Import necessary packages
import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import cv2

from distutils.version import StrictVersion
from collections import defaultdict
from io import StringIO
#from matplotlib import pyplot as plt
from PIL import Image

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")
from utils import ops as utils_ops

#if StrictVersion(tf.__version__) < StrictVersion('1.12.0'):
#  raise ImportError('Please upgrade your TensorFlow installation to v1.12.*.')

from utils import label_map_util

#from utils import visualization_utils as vis_util

# Size, in inches, of the output images.
#IMAGE_SIZE = (12, 8)

# Helper function to load image to numpy array for further processing
def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)

# Helper function to detect objects in single image
def run_inference_for_single_image(image, graph):
  with graph.as_default():
    with tf.Session() as sess:
      # Get handles to input and output tensors
      ops = tf.get_default_graph().get_operations()
      all_tensor_names = {output.name for op in ops for output in op.outputs}
      tensor_dict = {}
      for key in [
          'num_detections', 'detection_boxes', 'detection_scores',
          'detection_classes', 'detection_masks'
      ]:
        tensor_name = key + ':0'
        if tensor_name in all_tensor_names:
          tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
              tensor_name)
      if 'detection_masks' in tensor_dict:
        # The following processing is only for single image
        detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
        detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
        # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
        real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
        detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
        detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
            detection_masks, detection_boxes, image.shape[1], image.shape[2])
        detection_masks_reframed = tf.cast(
            tf.greater(detection_masks_reframed, 0.5), tf.uint8)
        # Follow the convention by adding back the batch dimension
        tensor_dict['detection_masks'] = tf.expand_dims(
            detection_masks_reframed, 0)
      image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

      # Run inference
      output_dict = sess.run(tensor_dict,
                             feed_dict={image_tensor: image})

      # all outputs are float32 numpy arrays, so convert types as appropriate
      output_dict['num_detections'] = int(output_dict['num_detections'][0])
      output_dict['detection_classes'] = output_dict[
          'detection_classes'][0].astype(np.uint8)
      output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
      output_dict['detection_scores'] = output_dict['detection_scores'][0]
      if 'detection_masks' in output_dict:
        output_dict['detection_masks'] = output_dict['detection_masks'][0]
  return output_dict

# Get session for image inference
def get_sess_for_inference(detection_graph):
  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True
  sess = tf.Session(graph=detection_graph, config=config)

  return sess

# Get inference result
def get_inference_of_image(image, detection_graph, sess):
  # Change image from BGR to RGB
  image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

  # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
  image_np_expanded = np.expand_dims(image_rgb, axis=0)

  # Get tensor from graph
  image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
  detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
  detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
  detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
  num_detections = detection_graph.get_tensor_by_name('num_detections:0')

  (boxes, scores, classes, num) = sess.run(
            [detection_boxes, detection_scores,
             detection_classes, num_detections],
            feed_dict={image_tensor: image_np_expanded})

  scores = np.squeeze(scores)
  classes = np.squeeze(classes).astype(np.int32)

  return scores, classes

# Get detection graph & category index from specified model
def get_model_info(model):
    # What model to download.
    if (model == 1):
        MODEL_NAME = 'ssd_mobilenet_v1_coco_2018_01_28'
    elif (model == 2):
        MODEL_NAME = 'faster_rcnn_inception_resnet_v2_atrous_coco_2018_01_28'
    elif (model == 3):
        MODEL_NAME = 'faster_rcnn_resnet101_coco_2018_01_28'
    elif (model == 4):
        MODEL_NAME = 'faster_rcnn_inception_v2_coco_2018_01_28'
    elif (model == 5):
        MODEL_NAME = 'ssdlite_mobilenet_v2_coco_2018_05_09'
    elif (model == 6):
        MODEL_NAME = 'faster_rcnn_inception_resnet_v2_atrous_coco_11_06_2017'
    elif (model == 7):
        MODEL_NAME = 'faster_rcnn_resnet101_coco_11_06_2017'
    elif (model == 8):
        MODEL_NAME = 'ssd_mobilenet_v1_coco_11_06_2017'
    elif (model == 9):
        MODEL_NAME = 'ssd_inception_v2_coco_11_06_2017'
    elif (model == 10):
        MODEL_NAME = 'rfcn_resnet101_coco_11_06_2017'

    MODEL_FILE = MODEL_NAME + '.tar.gz'
    DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

    # Path to frozen detection graph. This is the actual model that is used for the object detection.
    PATH_TO_FROZEN_GRAPH = MODEL_NAME + '/frozen_inference_graph.pb'

    # List of the strings that is used to add correct label for each box.
    PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')
    
    # Download Model
    if not (os.path.isfile(MODEL_FILE) or os.path.isdir(MODEL_NAME)):
        #print("model file " + MODEL_FILE + " has not been downloaded yet. Will download and extract here")
        opener = urllib.request.URLopener()
        opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
        tar_file = tarfile.open(MODEL_FILE)
        for file in tar_file.getmembers():
            file_name = os.path.basename(file.name)
            if 'frozen_inference_graph.pb' in file_name:
                tar_file.extract(file, os.getcwd())
    
    # Load frozen Tensorflow model into memory
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
    
    # Load label map
    category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)
    
    return detection_graph, category_index

# Detect and save image (with detected objects) with specified model (graph and label index)
def detect_and_save_image_model(image_path, save_path, detection_graph, category_index):
  image = Image.open(image_path)
  # the array based representation of the image will be used later in order to prepare the
  # result image with boxes and labels on it.
  image_np = load_image_into_numpy_array(image)
  # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
  image_np_expanded = np.expand_dims(image_np, axis=0)

  # Actual detection.
  output_dict = run_inference_for_single_image(image_np_expanded, detection_graph)
  # Visualization of the results of a detection.
  '''
  vis_util.visualize_boxes_and_labels_on_image_array(
      image_np,
      output_dict['detection_boxes'],
      output_dict['detection_classes'],
      output_dict['detection_scores'],
      category_index,
      instance_masks=output_dict.get('detection_masks'),
      use_normalized_coordinates=True,
      line_thickness=4)
  '''
  #plt.figure(figsize=IMAGE_SIZE)
  image_np_rgb = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
  cv2.imwrite(save_path, image_np_rgb)

# Get detection graph & category index from specified model
def get_final_model_info(path_to_graph, path_to_label):
    # Load frozen Tensorflow model into memory
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(path_to_graph, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

    # Load label map
    category_index = label_map_util.create_category_index_from_labelmap(path_to_label, use_display_name=True)

    return detection_graph, category_index

# Detect traffic light with specified model (graph and label index)
def get_detected_objects(image, detection_graph, category_index):
  # Change image from BGR to RGB
  image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

  # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
  image_np_expanded = np.expand_dims(image_rgb, axis=0)

  # Get inference of input image with detected objects
  output_dict = run_inference_for_single_image(image_np_expanded, detection_graph)

  return output_dict