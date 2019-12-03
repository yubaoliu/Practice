#ifndef _SEGNET_H_
#define _SEGNET_H_

#include "slamBase.h"

#include <caffe/caffe.hpp>
#include <glog/logging.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <algorithm>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>

using namespace caffe;

namespace RE_SLAM
{
    class Segnet
    {
    public:
        Segnet(const string &model_file,  const string &trained_file);

        cv::Mat Predict(const cv::Mat &img);

        cv::Mat Visualization(cv::Mat merged_output_image, string LUT_file);

    private:
        void SetMean(const string &mean_file);

        void WrapInputLayer(std::vector<cv::Mat> *input_channels);

        void Preprocess(const cv::Mat &img,
                        std::vector<cv::Mat> *input_channels);


    private:
        boost::shared_ptr<Net<float>> net_;
        cv::Size input_geometry_;
        int num_channels_;
};

} // namespace YB_SLAM
#endif