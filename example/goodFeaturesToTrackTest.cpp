#include <iostream>
#include "slamBase.h"
// #include "Frame.h"
// using namespace RE_SLAM;

// OpenCV 特征检测模块
#include <opencv2/features2d/features2d.hpp>
// #include <opencv2/nonfree/nonfree.hpp> // use this if you want to use SIFT or SURF
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
using namespace std;
// using namespace YB_SLAM;

int main(int argc, char **argv)
{
    //设置角点检测参数
    std::vector<cv::Point2f> corners;
    int max_corners = 1000;
    double quality_level = 0.01;
    double min_distance = 3.0;
    int block_size = 3;
    bool use_harris = false;
    double k = 0.04;
    cv::Mat image_color = cv::imread("../data/walk_rgb_1.png");
    cv::Mat image_gray;
    cv::cvtColor(image_color, image_gray, cv::COLOR_BGR2GRAY);

    cv::goodFeaturesToTrack(image_gray, corners, max_corners, quality_level, min_distance, cv::Mat(), block_size, true, k);
    
    //Criteria for termination of the iterative process of corner refinement.
    cv::TermCriteria criteria = cv::TermCriteria(
        cv::TermCriteria::MAX_ITER + cv::TermCriteria::EPS,
        20,
        0.003);

    cv::cornerSubPix(image_gray, corners, cv::Size(5, 5), cv::Size(-1, -1), criteria);

    for (int i = 0; i < corners.size(); i++)
    {
        cv::circle(image_color, corners[i], 1, cv::Scalar(0, 0, 255), 2, 8, 0);
    }
    cv::imshow("goodFeaturesToTrack", image_color);
    cv::waitKey(0);
    return 0;
}