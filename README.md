# Fine Tuning of Object Detection
## Objective
Train an object detection model to detect red traffic light. The frozed graph is then used for inference to control brake/throttle of an autonomous vehicle to stop the car at red traffic light and move under green light.

## Constraints
* The machine used for inference is poor, so the computation should be very efficient
* 


1. Add environment variables to `~/.bashrc` (or, `/home/workspace/.student_bashrc`)
```
export PATH=$PATH:/root/miniconda3/bin:/usr/local/cuda-9.0/bin
export LD_LIBRARY_PATH=/opt/carndcapstone/cuda-8.0/extras/CUPTI/lib64:/opt/carndcapstone/cuda-8.0/lib64:
export CUDA_HOME=/opt/carndcapstone/cuda-8.0
export PYTHONPATH=$PYTHONPATH:/home/workspace/object_detection/Tensorflow/models/research/slim
```

2. Make directory
```
cd /home/workspace
mkdir object_detection && cd "$_"
mkdir workspace
mkdir Tensorflow && cd "$_"
```

3. Install Tensorflow Models
```
git clone -b r1.13.0 https://github.com/tensorflow/models.git
cd models/research/
protoc object_detection/protos/*.proto --python_out=.
```
Make sure to have protoc from https://github.com/protocolbuffers/protobuf/releases

4. Install COCO API
```
git clone https://github.com/cocodataset/cocoapi.git
cd cocoapi/PythonAPI
make
cp -r pycocotools <PATH_TO_TF>/TensorFlow/models/research/
```

5. Prepare workspace
```
cd workspace
mkdir annotations
mkdir config
mkdir images
mkdir pre-trained-models
cp /home/workspace/object_detection/TensorFlow/models/research/object_detection/model_main.py .
```

6. Prepare labelled images

7. Prepare pipline config, label map
```
workspace
├─ annotations
│   ├─ label_map.pbtxt
│   ├─ sim_data.record
├─ config
│   ├─ faster_rcnn_resnet101_kitti.config
├─ images
│   ├─ test
│   └─ train
├─ pre-trained-model
│   ├─ faster_rcnn_resnet101_kitti_2018_01_28
```

## When running training
8. Execute environment setup script
```
cd /home/workspace/object_detection/
source setup_env.sh
```

9. Run training
```
python model_main.py --alsologtostderr --model_dir=/home/backups/training --pipeline_config_path=config/faster_rcnn_resnet101_kitti.config
```

To check the progress, run the following and access to localhost:6006
```
tensorboard --logdir=/home/backups/training/
```