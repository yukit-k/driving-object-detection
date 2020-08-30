#!/bin/bash

export WS_PATH=/home/workspace/object_detection

conda create -n od-py37 pip python=3.7
conda init bash
source ~/.bashrc
conda activate od-py37
pip install --upgrade tensorflow-gpu==1.14

conda install pillow lxml jupyter matplotlib opencv cython pandas
cd $WS_PATH/Tensorflow/models/research
pip install .
pip install labelImg

# Some modules required avoid errors
pip install absl-py gast==0.2.2 pycocotools

# Create training directory - this is deleted everytime you reconnect
export LOG_PATH=/home/backups/training
mkdir $LOG_PATH

cd $WS_PATH/workspace
