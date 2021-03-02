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

#include <string>
#include <pluginlib/class_list_macros.h>
#include "opencl_caffe/detector_gpu.h"
#include "opencl_caffe/nodelet.h"

PLUGINLIB_EXPORT_CLASS(opencl_caffe::Nodelet, nodelet::Nodelet)

namespace opencl_caffe
{
void Nodelet::onInit()
{
  ros::NodeHandle pnh = getPrivateNodeHandle();
  opencl_caffe::DetectorConfig config;
  if (!pnh.getParam("net_config_path", config.config))
  {
    ROS_WARN("param net_cfg_path not set, use default");
  }
  if (!pnh.getParam("weights_path", config.model))
  {
    ROS_WARN("param weights_path not set, use default");
  }
  if (!pnh.getParam("labels_path", config.classes))
  {
    ROS_WARN("param labels_path not set, use default");
  }


  loadResources(config);
  pub_ = getNodeHandle().advertise<object_msgs::ObjectsInBoxes>("inference", 1);
}

void Nodelet::cbImage(const sensor_msgs::ImagePtr image_msg)
{
  object_msgs::ObjectsInBoxes objects;
  if (detector_->runInference(image_msg, objects))
  {
    pub_.publish(objects);
  }
  else
  {
    ROS_ERROR("Inference failed.");
  }
}

void Nodelet::loadResources(const opencl_caffe::DetectorConfig& config)
{
  detector_.reset(new DetectorGpu());
  sub_.shutdown();

  if (detector_->loadResources(config))
  {
    sub_ = getNodeHandle().subscribe("/usb_cam/image_raw", 1, &Nodelet::cbImage, this);
  }
  else
  {
    ROS_FATAL("Load resource failed.");
    ros::shutdown();
  }
}
}  // namespace opencl_caffe
