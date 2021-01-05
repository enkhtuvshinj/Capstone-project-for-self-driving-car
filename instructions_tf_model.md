## Installing and training TensorFlow Object Detection API

This instruction covers how to install Tensorflow 1 Object Detection API on Windows 10 using Anaconda.

## Installation of required applications

1. Download and install [Visual Studio 2017](https://visualstudio.microsoft.com/vs/older-downloads/) that is required for compiling OpenCV.
2. Download and install [Anaconda3](https://www.anaconda.com/products/individual).
3. Download and install [CMake](https://cmake.org/).
4. Download [Protocol Buffer](https://github.com/protocolbuffers/protobuf/releases) that is required for the Tensorflow Object Detection API. Extract zipped exe to an arbitrary folder `<PROTOC_PATH>`.

## Preparing workspace on Anaconda

1. Open Anaconda Navigator and run cmd from the navigator. Run below commands to create new environment on Anaconda3.
```
$  conda create --name tf1_models python
$  conda activate tf1_models
$  conda install python=3.6.12
```
2. Install dependencies on newly-created environment by running below command from cmd.
```
$  pip install tensorflow=1.15 Cython contextlib2 pillow lxml jupyter matplotlib opencv
```
3. Run below commands to download the Tensorflow Object Detection API.
```
$  cd C:\Users\<YOUR_USERNAME>\anaconda3\envs\tf1_models\Lib\site-packages\tensorflow
$  git clone https://github.com/tensorflow/models
```
4. Execute protobuf compiler
```
$  cd C:\Users\<YOUR_USERNAME>\anaconda3\envs\tf1_models\Lib\site-packages\tensorflow\research 
$  <PROTOC_PATH>\bin\protoc.exe object_detection/protos/*.proto --python_out=.
```
5. Set the PYTHONPATH environment variable to the research, object_detection, and slim folders.
```
C:\Users\<YOUR_USERNAME>\anaconda3\envs\tf1_models\Lib\site-packages\tensorflow\research
C:\Users\<YOUR_USERNAME>\anaconda3\envs\tf1_models\Lib\site-packages\tensorflow\research\slim
C:\Users\<YOUR_USERNAME>\anaconda3\envs\tf1_models\Lib\site-packages\tensorflow\research\object_detection
```
6. Install the Tensorflow Object Detection API
```
$  cd C:\Users\<YOUR_USERNAME>\anaconda3\envs\tf1_models\Lib\site-packages\tensorflow\research 
$  cp object_detection/packages/tf1/setup.py .
$  python -m pip install .
```

7. Test whether Tensorflow Object Detection API works. If test results looks OK, then next step.
```
$  cd C:\Users\<YOUR_USERNAME>\anaconda3\envs\tf1_models\Lib\site-packages\tensorflow\research 
$  python object_detection/builders/model_builder_test.py
```

## Preparing the Dataset

Basically, preparing the dataset is a lot of manual works. We don't talk about how to prepare it, but if you are really curious I would recommend you read [TensorFlow OD API tutorial](https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/training.html).

## Configuring training

Let's assume that we already have a training, validation and test datasets. Now it is time to train a pre-trained model, MobileNet SSD v1 using datasets.
1. Download [Pre-Trained Model](http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_2017_11_17.tar.gz) and extract it to your new anaconda environment `<PATH_TO_MODEL>` that was created above.
2. We need to create `ssd_mobilenet_v1_coco.config` file based on our datasets and tuning parameters (training steps, learning_rate, image resize etc.) using a config in `C:\Users\<YOUR_USERNAME>\anaconda3\envs\tf1_models\Lib\site-packages\tensorflow\research\object_detection\configs\tf2`.
2. Train the model using modified config and datasets.
```
$  python object_detection\model_main.py --pipeline_config_path=<PATH_TO_MODEL>\ssd_mobilenet_v1_coco.config --train_dir=<PATH_TO_DATASET>\training_dataset
```
3. Once training is done, it produces a model check point file `model.ckpt-XXX`. Save the inference (trained model) using check point file.
```
$  python object_detection\export_inference_graph.py --pipeline_config_path=<PATH_TO_MODEL>\ssd_mobilenet_v1_coco.config --trained_checkpoint_prefix=<PATH_TO_DATASET>\training_dataset\model.ckpt-XXX --output_directory=<PATH_TO_MODEL>\model_frozen\
```

Now we have new trained model in `<PATH_TO_MODEL>\model_frozen\` folder. To know how to use the trained model, please look at tutorial `object_detection_tutorial.ipynb` in `C:\Users\<YOUR_USERNAME>\anaconda3\envs\tf1_models\Lib\site-packages\tensorflow\research\object_detection\colab_tutorials`.