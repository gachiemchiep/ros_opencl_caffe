/*
 * Copyright (c) 2017 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef OPENCL_CAFFE_DETECTOR_H
#define OPENCL_CAFFE_DETECTOR_H

#include <string>
#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <image_transport/image_transport.h>
#include <object_msgs/ObjectsInBoxes.h>

namespace opencl_caffe
{
/**
 * A structure to store configuration for Detector
*/
struct DetectorConfig
{
  std::string model;      // Neural network weights file path
  std::string config;     // Network configuration file path
  std::string classes;    // File path of labels of network output classes
  cv::Scalar mean = cv::Scalar(0, 0, 0);        // "Preprocess input image by subtracting mean values. 
  // default value for yolo : https://github.com/arunponnusamy/object-detection-opencv/issues/5
  float scale = 0.00392;  // "Preprocess input image by multiplying on a scale factor."
  float thr = 0.5;        // Confidence threshold.
  float nms = 0.4;        // Non-maximum suppression threshold.
  // OpenCV use BGR
  bool swapRB = true;     // Indicate that model works with RGB input images instead BGR ones.
  cv::Size inSize = cv::Size(416, 416);      // Preprocess input image that has size (w, h)
  int backend = cv::dnn::DNN_BACKEND_CUDA;  // Use cuda as default backend
  int target = cv::dnn::DNN_TARGET_CUDA;    // use cuda 32 as default target
};

/**
 * postprocess result
*/
struct DetectorRet
{
  std::vector<int> classIds;
  std::vector<float> confidences;
  std::vector<cv::Rect> boxes;
};

/** @class Detector
 * @brief Base class for detecting.
 * This class define a common interface of nueral network inference.
 * 1. Load all resources need by a network
 * 2. Run inference
 */
class Detector
{
public:
  /** Default destructor */
  virtual ~Detector() = default;
  /**
   * Load resources from file, construct a caffe Net object.
   *
   * @param[in]   net_cfg   Network configuration file path
   * @param[in]   weights   Neural network weights file path
   * @param[in]   labels    File path of labels of network output classes
   * @return     Status of load resources, true for success or false for failed
   */
  virtual int loadResources(const opencl_caffe::DetectorConfig& config) = 0;
  /**
   * Public interface of running inference to infer all objects in image.
   *
   * @param[in]   image_msg   image message subscribed from camera
   * @param[out]  objects     objects inferred
   * @return    Status of run inference, true for success or false for failed
   */
  virtual int runInference(const sensor_msgs::ImagePtr image_msg, object_msgs::ObjectsInBoxes& objects) = 0;
};
}  // namespace opencl_caffe

#endif  // OPENCL_CAFFE_DETECTOR_H
