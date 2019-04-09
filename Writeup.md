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
* [Notes](#Notes)

[//]: # (Image References)

[architecture]: ./imgs/architecture.png "System Architecture Diagram"

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

---
## Implementation

![alt text][architecture]

### Traffic light detection node

### Waypoint updater node

### DBW node

---
## Notes

### use_simulator_light_state

This is a defined parameter for specifying as an argument in command line to use the light state from simulator instead of detecting and classifying directly from camera image (for testing/debugging purpose). It is disabled by default and enabled by specifying as below:

```shell
$ roslaunch launch/styx.launch use_simulator_light_state:=true
```

### save_camera_image

This is a defined parameter for specifying as an argument in command line to save camera image for training/testing the model. It is disabled by default and enabled by specifying as below:

```shell
$ roslaunch launch/styx.launch save_camera_image:=true
```