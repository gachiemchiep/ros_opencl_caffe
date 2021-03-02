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

#ifndef OPENCL_CAFFE_DETECTOR_GPU_H
#define OPENCL_CAFFE_DETECTOR_GPU_H

#include <string>
#include <vector>

#include <opencv2/dnn.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <image_transport/image_transport.h>
#include <object_msgs/ObjectsInBoxes.h>
#include "opencl_caffe/detector.h"
#include "boost/date_time/posix_time/posix_time.hpp"
#include <boost/filesystem.hpp>

namespace opencl_caffe
{
/** @class DetectorGpu
 * @brief A implamentation of GPU detecting.
 * This class implament a caffe GPU object inference with 16 bit and 32 bit float point.
 * 1. Load resources need by caffe network
 * 2. Initialize the network (template for fp16 and fp32)
 * 3. Pre-process input image
 * 4. Infer the objects in image
 */
class DetectorGpu : public Detector
{
private:
  /** network */
  cv::dnn::Net net_;
  opencl_caffe::DetectorConfig config_;
  std::vector<std::string> outNames_; // name of output layer
  std::vector<std::string> classes_; //
  /**
   * Run inference to infer all objects in image.
   *
   * @param[in]       image_msg       Image messages as input
   * @param[out]       objects        Objects inferred as output
   * @return                          Status of inference, true for success or false for failed
   */
  int inference(const sensor_msgs::ImagePtr image_msg, object_msgs::ObjectsInBoxes& objects);

  void preprocess(const cv::Mat& frame, cv::dnn::Net& net, cv::Size inpSize, float scale,
                       const cv::Scalar& mean, bool swapRB);
  opencl_caffe::DetectorRet postprocess(cv::Mat& frame, const std::vector<cv::Mat>& outs, cv::dnn::Net& net, int backend);


public:
  /** Constructor*/
  DetectorGpu();
  /** Deafult deconstructor */
  ~DetectorGpu() = default;
  /**
   * Load resources from file, construct a dnn Net object using opencv dnn.
   *
   * @return     Status of load resources, true for success or false for failed
   */
  int loadResources(const opencl_caffe::DetectorConfig& config);
  /**
   * Public interface of running inference to infer all objects in image.
   *
   * @param[in]   image_msg   image message subscribed from camera
   * @param[out]  objects     objects inferred
   * @return    Status of run inference, true for success or false for failed
   */
  int runInference(const sensor_msgs::ImagePtr image_msg, object_msgs::ObjectsInBoxes& objects);
};
}  // namespace opencl_caffe

#endif  // OPENCL_CAFFE_DETECTOR_GPU_H
