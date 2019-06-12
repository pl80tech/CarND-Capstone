# **Capstone Project** 
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

## Writeup

This is my writeup for the project "Capstone" of Self Driving Car Nanadegree on Udacity.

---

### Contents

* [About Capstone Project](#About-Capstone-Project)
* [Project code](#Project-code)
* [Rubric Points](#Rubric-Points)
* [Code compilation](#Code-compilation)
* [Implementation](#Implementation)
	* [Traffic light detection node](#Traffic-light-detection-node)
	* [Waypoint updater node](#Waypoint-updater-node)
	* [DBW node](#DBW-node)
* [Model training approach](#Model-training-approach)
* [Model training and testing](#Model-training-and-testing)
* [Dataset preparation](#Dataset-preparation)
* [Get model file](#Get-model-file)
* [Output video](#Output-video)
* [Notes](#Notes)

[//]: # (Image References)

[architecture]: ./imgs/architecture.png "System Architecture Diagram"
[highway_sim_light_state]: ./video/highway_use_simulator_light_state.gif "Driving on highway using simulator light state"

---
## About Capstone Project

The goals / steps of this project are the following:

* The goal of this project is to program a Real Self-Driving Car on ROS (Robot Operating System) combining the technologies having been taught throughout the Nanodegree program.
* More detail explanation can be found in [README](https://github.com/pl80tech/CarND-Capstone/blob/master/README.md)

---
## Project code

Here is my working repository for this project:

https://github.com/pl80tech/CarND-Capstone.git

It is imported and frequently updated (cherry-pick or merge) from below original repository:

https://github.com/udacity/CarND-Capstone.git

---
## Rubric Points

Here are the [Rubric Points](https://review.udacity.com/#!/rubrics/1969/view) which need to be addressed to meet the requirements of this project.

---
## Code compilation

The implemented code can be compiled successfully by catkin_make. Refer to [/ros/catkin_make_log.txt](https://github.com/pl80tech/CarND-Capstone/blob/master/ros/catkin_make_log.txt) for the detail build log.

---
## Implementation

![alt text][architecture]

### Traffic light detection node

### Waypoint updater node

### DBW node

---
## Dataset preparation

The dataset for training and testing the model are uploaded to Google Drive and can be downloaded by following script: [dataset_prepare.py](https://github.com/pl80tech/CarND-Capstone/blob/master/ros/src/tl_detector/dataset_prepare.py). Here is an example of downloading dataset#1.

```shell
$ python dataset_prepare.py 1
```

---
## Model training approach

I have considered and tried to use below approaches to train the models for detecting and classifying traffic lights.

* Approach 1 (straight forward): use an NVIDIA-model based CNN architecture and train the model from scratch using the dataset collecting from camera images
* Approach 2: use existed/pretrained tensorflow models for object detection and retrain for only classifying traffic lights
* Approach 3: use 2 models - one for detecting/abstracting traffic light from camera image and one for classifying the detected traffic light

Due to time limitation, my submission is based on approach#1.

---
## Model training and testing

I implemented following script [train_classifier_model.py](https://github.com/pl80tech/CarND-Capstone/blob/master/ros/src/tl_detector/model/train_classifier_model.py) to create the model and necessary helper functions for training and testing. Then I trained the model with multiple combinations of setting and hyperparameters for tuning the best final model (see [tune_classifier_model.ipynb](https://github.com/pl80tech/CarND-Capstone/blob/master/ros/src/tl_detector/model/tune_classifier_model.ipynb) for more detail). I also created [test_trained_classifier.py](https://github.com/pl80tech/CarND-Capstone/blob/master/ros/src/tl_detector/model/test_trained_classifier.py) to run the trained classfication model for all images in specified path for quick verification.

---
## Get model file

The trained models for detection and classification (big size) are uploaded in Google Drive and can be downloaded by following script: [get_final_model.py](https://github.com/pl80tech/CarND-Capstone/blob/master/ros/src/tl_detector/get_final_model.py). Here is an example of downloading model#2.

```shell
$ python get_final_model.py 2
```

---
## Output video

Here is the video when driving on simulator using my trained model. The car can detect the traffic lights correctly and control its speed to drive through the highway autonomously.

---
## Notes

### tl_classification_config.yaml

This is a configuration file to customize the parameters for detecting and classifying the traffic lights without recompiling for quick confirmation and easy debug.

* path_to_graph: path to the frozen graph of the model
* path_to_label: path to the label map of the model
* detection_threshold: threshold to select the detected result for classification
* skip_interval: interval to skip processing the images from camera

### use_simulator_light_state

This is a defined parameter for specifying as an argument in command line to use the light state from simulator instead of detecting and classifying directly from camera image (for testing/debugging purpose). It is disabled by default and enabled by specifying as below:

```shell
$ roslaunch launch/styx.launch use_simulator_light_state:=true
```
Here is the simulation video on highway using simulator's light state in which the car can navigate through the traffic lights successfully. Click on the thumbnail animated gif to view the video directly on Youtube or click on the hyperlink to download the video on Github.

| Link on Github | [highway_use_simulator_light_state.mp4](https://github.com/pl80tech/CarND-Capstone/blob/master/video/highway_use_simulator_light_state.mp4) |
|:--------------:|:---------------:|
| Link on Youtube | [![alt text][highway_sim_light_state]](https://www.youtube.com/watch?v=5-mSSGskBSc) |

### save_camera_image

This is a defined parameter for specifying as an argument in command line to save camera image for training/testing the model. It is disabled by default and enabled by specifying as below:

```shell
$ roslaunch launch/styx.launch save_camera_image:=true
```

### save_inference_image

This is a defined parameter for specifying as an argument in command line to save inference image for easy comparison. It is disabled by default and enabled by specifying as below:

```shell
$ roslaunch launch/styx.launch save_camera_image:=true save_inference_image:=true
```