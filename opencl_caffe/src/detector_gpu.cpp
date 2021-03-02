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
#include <vector>
#include <cv_bridge/cv_bridge.h>
#include <ros/package.h>
#include "opencl_caffe/detector_gpu.h"

namespace opencl_caffe
{

inline void DetectorGpu::preprocess(const cv::Mat& frame, cv::dnn::Net& net, cv::Size inpSize, float scale,
                       const cv::Scalar& mean, bool swapRB)
{
    static cv::Mat blob;
    // Create a 4D blob from a frame.
    if (inpSize.width <= 0) inpSize.width = frame.cols;
    if (inpSize.height <= 0) inpSize.height = frame.rows;
    cv::dnn::blobFromImage(frame, blob, 1.0, inpSize, cv::Scalar(), swapRB, false, CV_8U);

    // Run a model.
    net.setInput(blob, "", scale, mean);
    if (net.getLayer(0)->outputNameToIndex("im_info") != -1)  // Faster-RCNN or R-FCN
    {
        resize(frame, frame, inpSize);
        cv::Mat imInfo = (cv::Mat_<float>(1, 3) << inpSize.height, inpSize.width, 1.6f);
        net.setInput(imInfo, "im_info");
    }
}

opencl_caffe::DetectorRet DetectorGpu::postprocess(cv::Mat& frame, const std::vector<cv::Mat>& outs, cv::dnn::Net& net, int backend)
{
    static std::vector<int> outLayers = net.getUnconnectedOutLayers();
    static std::string outLayerType = net.getLayer(outLayers[0])->type;

    std::vector<int> classIds;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;
    if (outLayerType == "DetectionOutput")
    {
        // Network produces output blob with a shape 1x1xNx7 where N is a number of
        // detections and an every detection is a vector of values
        // [batchId, classId, confidence, left, top, right, bottom]
        CV_Assert(outs.size() > 0);
        for (size_t k = 0; k < outs.size(); k++)
        {
            float* data = (float*)outs[k].data;
            for (size_t i = 0; i < outs[k].total(); i += 7)
            {
                float confidence = data[i + 2];
                if (confidence > this->config_.thr)
                {
                    int left   = (int)data[i + 3];
                    int top    = (int)data[i + 4];
                    int right  = (int)data[i + 5];
                    int bottom = (int)data[i + 6];
                    int width  = right - left + 1;
                    int height = bottom - top + 1;
                    if (width <= 2 || height <= 2)
                    {
                        left   = (int)(data[i + 3] * frame.cols);
                        top    = (int)(data[i + 4] * frame.rows);
                        right  = (int)(data[i + 5] * frame.cols);
                        bottom = (int)(data[i + 6] * frame.rows);
                        width  = right - left + 1;
                        height = bottom - top + 1;
                    }
                    classIds.push_back((int)(data[i + 1]) - 1);  // Skip 0th background class id.
                    boxes.push_back(cv::Rect(left, top, width, height));
                    confidences.push_back(confidence);
                }
            }
        }
    }
    else if (outLayerType == "Region")
    {
        for (size_t i = 0; i < outs.size(); ++i)
        {
            // Network produces output blob with a shape NxC where N is a number of
            // detected objects and C is a number of classes + 4 where the first 4
            // numbers are [center_x, center_y, width, height]
            float* data = (float*)outs[i].data;
            for (int j = 0; j < outs[i].rows; ++j, data += outs[i].cols)
            {
                cv::Mat scores = outs[i].row(j).colRange(5, outs[i].cols);
                cv::Point classIdPoint;
                double confidence;
                minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);
                if (confidence > this->config_.thr)
                {
                    int centerX = (int)(data[0] * frame.cols);
                    int centerY = (int)(data[1] * frame.rows);
                    int width = (int)(data[2] * frame.cols);
                    int height = (int)(data[3] * frame.rows);
                    int left = centerX - width / 2;
                    int top = centerY - height / 2;

                    classIds.push_back(classIdPoint.x);
                    confidences.push_back((float)confidence);
                    boxes.push_back(cv::Rect(left, top, width, height));
                }
            }
        }
    }
    else {
        ROS_ERROR("Unknown output layer type");
        ROS_ERROR(outLayerType.c_str());
    }
    // NMS is used inside Region layer only on DNN_BACKEND_OPENCV for another backends we need NMS in sample
    // or NMS is required if number of outputs > 1
    if (outLayers.size() > 1 || (outLayerType == "Region" && backend != cv::dnn::DNN_BACKEND_OPENCV))
    {
        std::map<int, std::vector<size_t> > class2indices;
        for (size_t i = 0; i < classIds.size(); i++)
        {
            if (confidences[i] >= this->config_.thr)
            {
                class2indices[classIds[i]].push_back(i);
            }
        }
        std::vector<cv::Rect> nmsBoxes;
        std::vector<float> nmsConfidences;
        std::vector<int> nmsClassIds;
        for (std::map<int, std::vector<size_t> >::iterator it = class2indices.begin(); it != class2indices.end(); ++it)
        {
            std::vector<cv::Rect> localBoxes;
            std::vector<float> localConfidences;
            std::vector<size_t> classIndices = it->second;
            for (size_t i = 0; i < classIndices.size(); i++)
            {
                localBoxes.push_back(boxes[classIndices[i]]);
                localConfidences.push_back(confidences[classIndices[i]]);
            }
            std::vector<int> nmsIndices;
            cv::dnn::NMSBoxes(localBoxes, localConfidences, this->config_.thr, this->config_.nms, nmsIndices);
            for (size_t i = 0; i < nmsIndices.size(); i++)
            {
                size_t idx = nmsIndices[i];
                nmsBoxes.push_back(localBoxes[idx]);
                nmsConfidences.push_back(localConfidences[idx]);
                nmsClassIds.push_back(it->first);
            }
        }
        boxes = nmsBoxes;
        classIds = nmsClassIds;
        confidences = nmsConfidences;
    }

  opencl_caffe::DetectorRet ret;
  ret.boxes = boxes;
  ret.classIds = classIds;
  ret.confidences = confidences;

  return ret;

}

int DetectorGpu::loadResources(const opencl_caffe::DetectorConfig& config)
{
  if (!boost::filesystem::exists(config.model) || !boost::filesystem::exists(config.config) || !boost::filesystem::exists(config.classes))
  {
    ROS_ERROR("Network configuration file or weights file not found!");
    return false;
  }

  // Load labels
  std::ifstream fs(config.classes);
  std::string label_name;
  while (std::getline(fs, label_name))
  {
    this->classes_.push_back(label_name);
  }

  // init network

  this->net_ = cv::dnn::readNet(config.model, config.config);
  this->net_.setPreferableBackend(config.backend);
  this->net_.setPreferableTarget(config.target);

  // store other configuration
  this->config_ = config;
  this->outNames_ = this->net_.getUnconnectedOutLayersNames();

  // warmingup network 
  cv::Mat image = cv::imread(ros::package::getPath("opencl_caffe") + "/resources/cat.jpg");
  this->preprocess(image, this->net_, this->config_.inSize, this->config_.scale, this->config_.mean, this->config_.swapRB);
  std::vector<cv::Mat> outs;
  this->net_.forward(outs, this->outNames_);

  ROS_INFO("Load resources completely!");
  return true;
}

int DetectorGpu::inference(const sensor_msgs::ImagePtr image_msg, object_msgs::ObjectsInBoxes& objects)
{
  try
  {
    cv::Mat image;

    boost::posix_time::ptime start = boost::posix_time::microsec_clock::local_time();

    opencl_caffe::DetectorRet ret;
    cv::cvtColor(cv_bridge::toCvShare(image_msg, "rgb8")->image, image, cv::COLOR_RGB2BGR);

    preprocess(image, this->net_, this->config_.inSize, this->config_.scale, this->config_.mean, this->config_.swapRB);
    std::vector<cv::Mat> outs;
    this->net_.forward(outs, this->outNames_);
    ret = postprocess(image, outs, this->net_, this->config_.backend);

    boost::posix_time::ptime end = boost::posix_time::microsec_clock::local_time();
    boost::posix_time::time_duration msdiff = end - start;

    const int num_det = ret.classIds.size();

    for (int k = 0; k < num_det; k++)
    {
      object_msgs::ObjectInBox object_in_box;
      object_in_box.object.object_name = ret.classIds[k];
      object_in_box.object.probability = ret.confidences[k];
      object_in_box.roi.x_offset = ret.boxes[k].x;
      object_in_box.roi.y_offset = ret.boxes[k].y;
      object_in_box.roi.height = ret.boxes[k].height;
      object_in_box.roi.width = ret.boxes[k].width;
      objects.objects_vector.push_back(object_in_box);
    }

    objects.header = image_msg->header;
    objects.inference_time_ms = msdiff.total_milliseconds();
    return true;
  }
  catch (cv_bridge::Exception& e)
  {
    ROS_ERROR("Could not covert from '%s' to 'rgb8'.", image_msg->encoding.c_str());
    return false;
  }
}

int DetectorGpu::runInference(const sensor_msgs::ImagePtr image_msg, object_msgs::ObjectsInBoxes& objects)
{
  return inference(image_msg, objects);
}
}  // namespace opencl_caffe
