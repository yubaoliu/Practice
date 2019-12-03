#include "Segnet.h"

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

namespace RE_SLAM
{

Segnet::Segnet(const string &model_file,
               const string &trained_file)
{
    Caffe::set_mode(Caffe::GPU);

    /* Load the network. */
    net_.reset(new Net<float>(model_file, TEST));
    net_->CopyTrainedLayersFrom(trained_file);

    CHECK_EQ(net_->num_inputs(), 1) << "Network should have exactly one input.";
    CHECK_EQ(net_->num_outputs(), 1) << "Network should have exactly one output.";

    Blob<float> *input_layer = net_->input_blobs()[0];
    num_channels_ = input_layer->channels();
    CHECK(num_channels_ == 3 || num_channels_ == 1)
        << "Input layer should have 1 or 3 channels.";
    input_geometry_ = cv::Size(input_layer->width(), input_layer->height());
}

cv::Mat Segnet::Predict(const cv::Mat &img)
{
    Blob<float> *input_layer = net_->input_blobs()[0];
    input_layer->Reshape(1, num_channels_,
                         input_geometry_.height, input_geometry_.width);
    /* Forward dimension change to all layers. */
    net_->Reshape();

    std::vector<cv::Mat> input_channels;
    WrapInputLayer(&input_channels);

    Preprocess(img, &input_channels);

    struct timeval time;
    gettimeofday(&time, NULL); // Start Time
    long totalTime = (time.tv_sec * 1000) + (time.tv_usec / 1000);
    //std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now(); //Just for time measurement

    net_->Forward();

    gettimeofday(&time, NULL); //END-TIME
    totalTime = (((time.tv_sec * 1000) + (time.tv_usec / 1000)) - totalTime);
    std::cout << "Processing time = " << totalTime << " ms" << std::endl;

    //std::chrono::steady_clock::time_point end= std::chrono::steady_clock::now();
    //std::cout << "Processing time = " << (std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count())/1000000.0 << " sec" <<std::endl; //Just for time measurement

    /* Copy the output layer to a std::vector */
    Blob<float> *output_layer = net_->output_blobs()[0];

    std::cout << "output_blob(n,c,h,w) = " << output_layer->num() << ", " << output_layer->channels() << ", "
              << output_layer->height() << ", " << output_layer->width() << std::endl;

//     cv::Mat merged_output_image = cv::Mat(output_layer->height(), output_layer->width(), CV_32F, const_cast<float *>(output_layer->cpu_data()));
//     //merged_output_image = merged_output_image/255.0;

//     merged_output_image.convertTo(merged_output_image, CV_8U);
//     cv::cvtColor(merged_output_image.clone(), merged_output_image, CV_GRAY2BGR);

//     return merged_output_image;

    int width = output_layer->width();
    int height = output_layer->height();
    int channels = output_layer->channels();

    cv::Mat class_each_row(channels, width * height, CV_32FC1, const_cast<float *>(output_layer->cpu_data()));
    class_each_row = class_each_row.t();

    cv::Point maxId;
    double maxValue;
    cv::Mat prediction_map(height, width, CV_8UC1);

    for (int i = 0; i < class_each_row.rows; i++)
    {
        cv::minMaxLoc(class_each_row.row(i), 0, &maxValue, 0, &maxId);
        prediction_map.at<uchar>(i) = maxId.x;
    }
    return prediction_map;
}

cv::Mat Segnet::Visualization(cv::Mat predict_img, string LUT_file)
{
    cv::Mat label_colours = cv::imread(LUT_file, 1);
    cv::Mat output_image = predict_img.clone();
    cv::cvtColor(predict_img, output_image, cv::COLOR_GRAY2BGR);
    LUT(output_image, label_colours, output_image);

    return output_image;
}


/* Wrap the input layer of the network in separate cv::Mat objects
 * (one per channel). This way we save one memcpy operation and we
 * don't need to rely on cudaMemcpy2D. The last preprocessing
 * operation will write the separate channels directly to the input
 * layer. */
void Segnet::WrapInputLayer(std::vector<cv::Mat> *input_channels)
{
    Blob<float> *input_layer = net_->input_blobs()[0];

    int width = input_layer->width();
    int height = input_layer->height();
    float *input_data = input_layer->mutable_cpu_data();
    for (int i = 0; i < input_layer->channels(); ++i)
    {
        cv::Mat channel(height, width, CV_32FC1, input_data);
        input_channels->push_back(channel);
        input_data += width * height;
    }
}

void Segnet::Preprocess(const cv::Mat &img,
                        std::vector<cv::Mat> *input_channels)
{
    /* Convert the input image to the input image format of the network. */
    cv::Mat sample;
    if (img.channels() == 3 && num_channels_ == 1)
        cv::cvtColor(img, sample, cv::COLOR_BGR2GRAY);
    else if (img.channels() == 4 && num_channels_ == 1)
        cv::cvtColor(img, sample, cv::COLOR_BGRA2GRAY);
    else if (img.channels() == 4 && num_channels_ == 3)
        cv::cvtColor(img, sample, cv::COLOR_BGRA2BGR);
    else if (img.channels() == 1 && num_channels_ == 3)
        cv::cvtColor(img, sample, cv::COLOR_GRAY2BGR);
    else
        sample = img;

    cv::Mat sample_resized;
    if (sample.size() != input_geometry_)
        cv::resize(sample, sample_resized, input_geometry_);
    else
        sample_resized = sample;

    cv::Mat sample_float;
    if (num_channels_ == 3)
        sample_resized.convertTo(sample_float, CV_32FC3);
    else
        sample_resized.convertTo(sample_float, CV_32FC1);

    /* This operation will write the separate BGR planes directly to the
   * input layer of the network because it is wrapped by the cv::Mat
   * objects in input_channels. */
    cv::split(sample_float, *input_channels);

    CHECK(reinterpret_cast<float *>(input_channels->at(0).data) == net_->input_blobs()[0]->cpu_data())
        << "Input channels are not wrapping the input layer of the network.";
}

} // namespace RE_SLAM