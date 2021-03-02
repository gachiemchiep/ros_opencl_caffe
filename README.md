
# How to start

## Compile

```bash




compile newest opencv -> we need the DNN modules 
# in the python2 environment
conda install fontconfig=2.13.1 pango=1.40
# goto : https://developer.nvidia.com/cuda-gpus
# to check the compute capability
# gtx 1060 = 6.1
cmake \
    -D CMAKE_INSTALL_PREFIX=/usr/local \
    -D OPENCV_EXTRA_MODULES_PATH=/home/jil/opt/opencv/opencv_contrib/modules \
    -D OPENCV_ENABLE_NONFREE=OFF \
    -D BUILD_NEW_PYTHON_SUPPORT=ON \
    -D BUILD_opencv_python3=ON \
    -D BUILD_opencv_python2=ON \
    -D INSTALL_PYTHON_EXAMPLES=ON \
    -D OPENCV_FORCE_PYTHON_LIBS=ON \
    -D PYTHON2_EXECUTABLE=/home/jil/opt/miniconda2/bin/python \
    -D PYTHON2_LIBRARY=/home/jil/opt/miniconda2/lib/libpython2.7.so \
    -D PYTHON2_INCLUDE_DIRS=/home/jil/opt/miniconda2/include/ \
    -D PYTHON2_NUMPY_INCLUDE_DIRS=/home/jil/opt/miniconda2/lib/python2.7/site-packages/numpy \
    -D PYTHON3_EXECUTABLE=/home/jil/opt/miniconda2/envs/py3.6/bin/python \
    -D PYTHON3_LIBRARY=/home/jil/opt/miniconda2/envs/py3.6/lib/libpython3.6m.so \
    -D PYTHON3_INCLUDE_DIRS=/home/jil/opt/miniconda2/envs/py3.6/include \
    -D PYTHON3_NUMPY_INCLUDE_DIRS=/home/jil/opt/miniconda2/envs/py3.6/lib/python3.6/site-packages/numpy \
    -D WITH_FFMPEG=1 \
    -D WITH_TIFF=ON \
    -D WITH_CUDA=ON \
    -D WITH_CUDNN=ON \
    -D OPENCV_DNN_CUDA=ON \
    -D ENABLE_FAST_MATH=1 \
    -D CUDA_FAST_MATH=1 \
    -D WITH_CUBLAS=1 \
    -D WITH_TBB=ON \
    -D WITH_OPENMP=ON \
    -D WITH_IPP=ON \
    -D CMAKE_BUILD_TYPE=RELEASE \
    -D BUILD_EXAMPLES=OFF \
    -D WITH_NVCUVID=ON \
    -D BUILD_DOCS=OFF \
    -D BUILD_PERF_TESTS=OFF \
    -D BUILD_TESTS=OFF \
    -D WITH_CSTRIPES=ON \
    -D WITH_OPENCL=ON \
    -D BUILD_opencv_cudacodec=OFF \
    -D CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-10.2 \
    -D CUDA_ARCH_BIN="6.1 7.0 7.5" -D CUDA_ARCH_PTX="" \
    ../opencv

# for python2
cp lib/cv2.so /home/jil/opt/miniconda2/lib/python2.7/site-packages/cv2.so 
# for python3
cp lib/python3/cv2.cpython-36m-x86_64-linux-gnu.so /home/jil/opt/miniconda2/envs/py3.6/lib/python3.6/site-packages/cv2.cpython-36m-x86_64-linux-gnu.so 

# Download yolo model from https://github.com/AlexeyAB/darknet#pre-trained-models
# yolov4.cfg,  yolov4.weights
# voc classes from : https://github.com/opencv/opencv/blob/master/samples/data/dnn/object_detection_classes_pascal_voc.txt

# Follow these 2 links to compile this project with lastest version of opencv (4.5)
# https://github.com/ros/catkin/issues/1004#issuecomment-481414097
# https://answers.ros.org/question/318146/using-opencv-dnn-module-with-ros/
# need to build cv_bridge and image transport plugins with opencv 4.5
git clone https://github.com/ros-perception/vision_opencv src/vision_opencv
git clone https://github.com/ros-perception/image_transport_plugins.git src/image_transport_plugins

catkin build cv_bridge
catkin build image_transport_plugins

source devel/setup.bash
catkin build opencl_caffe --force-cmake -G"Eclipse CDT4 - Unix Makefiles"

```

## Running the Demo

```bash


```









# ros_opencl_caffe
# Warning: This repo is deprecated. For latest ROS wrapper for Intel GPU, please refer to our project ros_openvino_toolkit. 
# See: https://github.com/intel/ros_openvino_toolkit!!

## Introduction
OpenCL Caffe([clCaffe](https://github.com/01org/caffe/tree/inference-optimize)) is an OpenCL backend for Caffe from Intel&reg;. With inference optimized by Intel&reg; OpenCL, clCaffe can be used in most scene in high performance, like objects inference.

This project is a ROS wrapper for OpenCL Caffe, providing following features:
* A ROS service for objects inference in a ROS image message
* A ROS publisher for objects inference in a ROS image message from a RGB camera
* Demo applications to show the capablilities of ROS service and publisher

## Prerequisite
* An x86_64 computer with Ubuntu 16.04
* ROS Kinetic
* RGB camera, e.g. Intel&reg; RealSense&trade;, Microsoft&reg; Kinect&trade; or standard USB camera

## Environment Setup
* Install ROS Kinetic Desktop-Full ([guide](http://wiki.ros.org/kinetic/Installation/Ubuntu))
* Create a catkin workspace ([guide](http://wiki.ros.org/catkin/Tutorials/create_a_workspace))
* Install clCaffe ([guide](https://github.com/01org/caffe/wiki/clCaffe))
* Create a symbol link in `/opt/clCaffe`
```Shell
  sudo ln -s <clCaffe-path> /opt/clCaffe
```
* Add clCaffe libraries to LD_LIBRARY_PATH.
```Shell
  export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/clCaffe/build/lib
```
* Install ROS package for different cameras as needed. e.g.
  1. Standard USB camera
  ```Shell
    sudo apt-get install ros-kinetic-usb-cam
  ```
  2. Intel&reg; RealSense&trade; camera
  - Install Intel&reg; RealSense&trade; SDK 2.0 ([guide](https://github.com/IntelRealSense/librealsense/blob/master/doc/distribution_linux.md)). Refer [here](https://github.com/IntelRealSense/librealsense) for more details about Intel&reg; RealSense&trade; SDK 2.0.
  - Install Intel&reg; RealSense&trade; ROS ([guide](https://github.com/intel-ros/realsense))
  ```Shell
    cd ~/catkin_ws/src
    git clone https://github.com/intel-ros/realsense.git
    cd  realsense
    git checkout 2.0.0
    cd ~/catkin_ws
    catkin_make
  ```
  3. Microsoft&reg; Kinect&trade; camera
  ```Shell
    sudo apt-get install ros-kinetic-openni-launch
  ```

## Building and Installation
```Shell
  cd ~/catkin_ws/src
  git clone https://github.com/intel/object_msgs
  git clone https://github.com/intel/ros_opencl_caffe
  cd ~/catkin_ws/
  catkin_make
  catkin_make install
  source install/setup.bash
```
Copy object label file to clCaffe installation location
```Shell
  cp ~/catkin_ws/src/ros_opencl_caffe/opencl_caffe/resources/voc.txt /opt/clCaffe/data/yolo/
```

## Running the Demo
### Inference
  1. Standard USB camera
```Shell
    roslaunch opencl_caffe_launch usb_cam_viewer.launch
```
  2. Intel&reg; RealSense&trade; camera
```Shell
    roslaunch opencl_caffe_launch realsense_viewer.launch
```
  3. Microsoft&reg; Kinect&trade; camera
```Shell
    roslaunch opencl_caffe_launch kinect_viewer.launch
```

### Service
```Shell
  roslaunch opencl_caffe_launch opencl_caffe_srv.launch
```

## Test
Use `rostest` for tests
```Shell
  source devel/setup.bash
  rostest opencl_caffe service.test
  rostest opencl_caffe detector.test
```

## Known Issues
* Only image messages supported in service demo
* Only test on RealSense D400 series camera, Microsoft Kinect v1 camera and Microsoft HD-300 USB camera

## TODO
* Support latest clCaffe

###### *Any security issue should be reported using process at https://01.org/security*
