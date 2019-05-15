#!/bin/sh
sudo apt-get install -y ros-kinetic-dbw-mkz-msgs
rosdep install --from-paths src --ignore-src --rosdistro=kinetic -y