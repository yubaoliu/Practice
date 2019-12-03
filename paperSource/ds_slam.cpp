#include <iostream>
#include "slamBase.h"
// #include "Frame.h"

// OpenCV 特征检测模块
#include <opencv2/features2d/features2d.hpp>
// #include <opencv2/nonfree/nonfree.hpp> // use this if you want to use SIFT or SURF
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/video/tracking.hpp>
using namespace std;
// using namespace YB_SLAM;

int main(int argc, char **argv)
{
    //设置角点检测参数
    std::vector<cv::Point2f> prePts, nextPts, prePts_tmp, nextPts_tmp, nextPtsInlier, prePtsInlier, nextPtsOutlier, prePtsOutlier;
    int max_corners = 1000;
    double quality_level = 0.01;
    double min_distance = 3.0;
    int block_size = 3;
    // bool use_harris = false;
    double k = 0.04;
    cv::Mat rgb1 = cv::imread("../data/walk_rgb_1.png");
    cv::Mat rgb2 = cv::imread("../data/walk_rgb_2.png");
    cv::Mat depth1 = cv::imread("../data/walk_depth_1.png", -1);
    cv::Mat depth2 = cv::imread("../data/walk_depth_2.png", -1);

    cv::Mat prevImg;
    cv::Mat nextImg;
    cv::cvtColor(rgb1, prevImg, cv::COLOR_BGR2GRAY);
    cv::cvtColor(rgb2, nextImg, cv::COLOR_RGB2GRAY);

    //The first step is to calculate optical flow pyramid to get the matched feature points in the current frame. Thencloud_viewer
    cv::goodFeaturesToTrack(prevImg, prePts, max_corners, quality_level, min_distance, cv::Mat(), block_size, true, k);

    //Criteria for termination of the iterative process of corner refinement.
    cv::TermCriteria criteria = cv::TermCriteria(
        cv::TermCriteria::MAX_ITER + cv::TermCriteria::EPS,
        20,
        0.003);

    cv::cornerSubPix(prevImg, prePts, cv::Size(5, 5), cv::Size(-1, -1), criteria);
    vector<uchar> status;
    vector<float> err;
    //status – output status vector (of unsigned chars); each element of the vector is set to 1 if the flow for the corresponding features has been found, otherwise, it is set to 0.

    cv::calcOpticalFlowPyrLK(prevImg, nextImg, prePts, nextPts, status, err, cv::Size(22, 22), 5, cv::TermCriteria(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 20, 0.01));

    //Remove Outlier
    // Then if the matched pair is too close to the edge of the image or the pixel difference of the 3×3 image block at the center of the matched pair is too large, the matched pair will be discarded.
    int limit_edge_corner = 5;
    double limit_of_check = 2120;
    for (int i = 0; i < status.size(); i++)
    {
        if (status[i] != 0)
        {
            int dx[10] = {-1, 0, 1, -1, 0, 1, -1, 0, 1};
            int dy[10] = {-1, -1, -1, 0, 0, 0, 1, 1, 1};
            int x1 = prePts[i].x, y1 = prePts[i].y;
            int x2 = nextPts[i].x, y2 = nextPts[i].y;
            if ((x1 < limit_edge_corner || x1 >= nextImg.cols - limit_edge_corner || x2 < limit_edge_corner || x2 >= nextImg.cols - limit_edge_corner || y1 < limit_edge_corner || y1 >= nextImg.rows - limit_edge_corner || y2 < limit_edge_corner || y2 >= nextImg.rows - limit_edge_corner))
            {
                status[i] = 0;
                continue;
            }
            double sum_check = 0;
            for (int j = 0; j < 9; j++)
                sum_check += abs(prevImg.at<uchar>(y1 + dy[j], x1 + dx[j]) - nextImg.at<uchar>(y2 + dy[j], x2 + dx[j]));
            if (sum_check > limit_of_check)
                status[i] = 0;
            if (status[i])
            {
                prePtsInlier.push_back(prePts[i]);
                nextPtsInlier.push_back(nextPts[i]);
            }
        }
    }

    //The third step is to find fundamental matrix by using RANSAC with the most inliers. Then calculate the epipolar line in the current frame by using the fundamental matrix. Finally, determine whether the distance from a matched point to its corresponding epipolar line is less than a certain threshold. If the distance is greater than the threshold, then the matched point will be determined to be moving.
    // F-Matrix
    cv::Mat mask = cv::Mat(cv::Size(1, 300), CV_8UC1);
    cv::Mat F = cv::findFundamentalMat(prePtsInlier, nextPtsInlier, mask, cv::FM_RANSAC, 0.1, 0.99);
    for (int i = 0; i < mask.rows; i++)
    {
        if (mask.at<uchar>(i, 0) == 0)
            ;
        else
        {
            // Circle(pre_frame, F_prepoint[i], 6, Scalar(255, 255, 0), 3);
            double A = F.at<double>(0, 0) * prePtsInlier[i].x + F.at<double>(0, 1) * prePtsInlier[i].y + F.at<double>(0, 2);
            double B = F.at<double>(1, 0) * prePtsInlier[i].x + F.at<double>(1, 1) * prePtsInlier[i].y + F.at<double>(1, 2);
            double C = F.at<double>(2, 0) * prePtsInlier[i].x + F.at<double>(2, 1) * prePtsInlier[i].y + F.at<double>(2, 2);
            double dd = fabs(A * nextPtsInlier[i].x + B * nextPtsInlier[i].y + C) / sqrt(A * A + B * B); //Epipolar constraints
            if (dd <= 0.1)
            {
                prePts_tmp.push_back(prePtsInlier[i]);
                nextPts_tmp.push_back(nextPtsInlier[i]);
            }
        }
    }
    prePtsInlier = prePts_tmp;
    nextPtsInlier = nextPts_tmp;

    //judge outliers
    for (int i = 0; i < prePts.size(); i++)
    {
        if (status[i] != 0)
        {
            double A = F.at<double>(0, 0) * prePts[i].x + F.at<double>(0, 1) * prePts[i].y + F.at<double>(0, 2);
            double B = F.at<double>(1, 0) * prePts[i].x + F.at<double>(1, 1) * prePts[i].y + F.at<double>(1, 2);
            double C = F.at<double>(2, 0) * prePts[i].x + F.at<double>(2, 1) * prePts[i].y + F.at<double>(2, 2);
            double dd = fabs(A * nextPts[i].x + B * nextPts[i].y + C) / sqrt(A * A + B * B);

            // Judge outliers
            double limit_dis_epi = 1;
            if (dd <= limit_dis_epi)
                continue;
            nextPtsOutlier.push_back(nextPts[i]);
        }
    }

    //visualize
    for (int i = 0; i < nextPtsOutlier.size(); i++)
    {
        cv::circle(rgb2, nextPtsOutlier[i], 1, cv::Scalar(0, 0, 255), 2, 8, 0);
    }

    cv::imshow("OutlierPoints", rgb2);
    cv::waitKey(0);

    //visualize inliers
    for (int i = 0; i < nextPtsInlier.size(); i++)
    {
        cv::circle(rgb2, nextPtsInlier[i], 1, cv::Scalar(0, 0, 255), 2, 8, 0);
    }

    cv::imshow("nextPtsInlier", rgb2);
    cv::waitKey(0);

    return 0;
}