#include "Segnet.h"

// #include <glog/logging.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <algorithm>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>

using namespace RE_SLAM;
void print_help()
{
    std::cerr << "Usage: EXE "
              << " \ndeploy.prototxt"
              << "\nnetwork.caffemodel"
              << " \nLUT (for example: pascal.png)"
              << std::endl;
}

int main(int argc, char **argv)
{
    ::google::InitGoogleLogging(argv[0]);
    if (argc != 4)
    {
        print_help();
        return 1;
    }

    //capture image
    cv::Mat image;
    cv::VideoCapture cap(0);
    if (!cap.isOpened())
    {
        cerr << "Open Camera Failed" << endl;
        return -1;
    }
    else
    {
        cout << "Open Camera Succeed" << endl;
    }

    //segnet config
    string model_file = argv[1];
    string trained_file = argv[2]; //for visualization
    string LUT_file = argv[3];
    Segnet classifier(model_file, trained_file);

    // Main loop
    int timeStamps = 0;
    cv::Mat predict_image;

    for (;; timeStamps++)
    {
        cap >> image;

        if (image.empty())
        {
            cerr << endl
                 << "Failed to load image at: " << endl;
            return 1;
        }
        CHECK(!image.empty()) << "Unable to decode image " << image;

        cv::Mat label = classifier.Predict(image);
        std::cout << "label: " << label.size() << endl;

        cv::Mat color_label = classifier.Visualization(label, LUT_file);
        cv::resize(label, label, cv::Size(image.cols, image.rows));
        cv::resize(color_label, color_label, cv::Size(image.cols, image.rows));

        // for (int row = 0; row < label.rows; row++)
        // {
        //     for (int col = 0; col < label.cols; col++)
        //     {
        //         // cout << (int)(label.at<cv::Vec3b>(row, col)[0]) << std::endl;

        //         if ((int)label.at<uchar>(row, col) == 15)
        //             std::cout << "------------People----------------" << std::endl;
        //         col = col+3;

        //     }
        // }

        cv::imshow("image", image);
        cv::imshow("predicted", color_label);
        cv::waitKey(1);
    }
}