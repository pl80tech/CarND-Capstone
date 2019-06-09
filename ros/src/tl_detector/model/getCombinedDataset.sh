#!/bin/sh

# Initial settings
datasets="1 2 3 4 5 6 7 8 9"

# Loop for all dataset
for dataset in $datasets
do
  echo "--------------------------------------"
  echo "Download dataset #$dataset"
  python ../dataset_prepare.py $dataset
done