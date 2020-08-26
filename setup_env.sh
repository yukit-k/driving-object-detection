#!/bin/bash

WS_PATH=/home/workspace/object_detection

conda create -n od-py37 pip python=3.7
conda init bash
source ~/.bashrc
conda activate od-py37
pip install --upgrade tensorflow-gpu==1.14

conda install pillow lxml jupyter matplotlib opencv cython pandas
cd $WS_PATH/TensorFlow/models/research
pip install .
pip install labelImg

# Some modules required avoid errors
pip install absl-py gast==0.2.2 pycocotools

mkdir /home/backups/training && cd "$_"
cp $WS_PATH/workspace/config/faster_rcnn_resnet101_kitti.config .

cd $WS_PATH/workspace
