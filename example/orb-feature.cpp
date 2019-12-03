#include <iostream>
#include <algorithm>  
#include <vector>
#include <boost/array.hpp>
#include <boost/algorithm/string.hpp>

// OpenCV 特征检测模块
#include <opencv2/features2d/features2d.hpp>
// #include <opencv2/nonfree/nonfree.hpp> // use this if you want to use SIFT or SURF
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/video/tracking.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/ml/ml.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d.hpp>

using namespace std;
using namespace cv;

const int MAX_FEATURES = 1000;

int main(int argc, char **argv)
{
    //walk
    // cv::Mat rgb1 = cv::imread("../data/walk/walk_rgb_1.png");
    // cv::Mat label1 = cv::imread("../data/walk/walk_rgb_1_label.png", cv::IMREAD_GRAYSCALE);
    // cv::Mat depth1 = cv::imread("../data/walk/walk_depth_1.png");

    // cv::Mat rgb2 = cv::imread("../data/walk/walk_rgb_2.png");
    // cv::Mat depth2 = cv::imread("../data/walk/walk_depth_2.png");
    // cv::Mat label2 = cv::imread("../data/walk/walk_rgb_2_label.png", cv::IMREAD_GRAYSCALE);

    //person or other dynamic objects in the monitor
    cv::Mat rgb1 = cv::imread("../data/lovelive/frame0059.jpg");
    // cv::Mat label1 = cv::imread("../data/lovelive/frame0058_label.png", cv::IMREAD_GRAYSCALE);

    cv::Mat rgb2 = cv::imread("../data/lovelive/frame0060.jpg");
    // cv::Mat label2 = cv::imread("../data/lovelive/frame0059_label.png", cv::IMREAD_GRAYSCALE);


    cv::resize(rgb1, rgb1, cv::Size(640, 480));
    // cv::resize(label1, label1,  cv::Size(640, 480));
    // cv::resize(depth1, depth1, cv::Size(640, 480));

    cv::resize(rgb2, rgb2, cv::Size(640, 480));
    // cv::resize(label2, label2, cv::Size(640, 480));
    // cv::resize(depth2, depth2, cv::Size(640, 480));

    cv::Mat prevImg;
    cv::Mat currentImg;
    cv::cvtColor(rgb1, prevImg, cv::COLOR_BGR2GRAY);
    cv::cvtColor(rgb2, currentImg, cv::COLOR_RGB2GRAY);
    

    if(prevImg.empty() || currentImg.empty() || prevImg.channels() != 1 || currentImg.channels() != 1)
    {
        cout<<"Input Image is nullptr or the image channels is not gray!" << endl;
        system("pause");
    }

    vector<cv::KeyPoint> kpts1, kpts2;
    cv::Mat desc1, desc2;

    Ptr<cv::Feature2D> orb = ORB::create(MAX_FEATURES, 1.2, 8, 31, 0, 2, 0, 31, 20);

    orb->detectAndCompute(prevImg, cv::Mat(), kpts1, desc1);
    orb->detectAndCompute(currentImg, cv::Mat(), kpts2, desc2);
    
    // cv::Mat show_feature_pre;
    // rgb1.copyTo(show_feature_pre);
    // for (int i = 0; i < kpts1.size(); i++)
    // {
    //     cv::circle(show_feature_pre, kpts1[i].pt, 1, cv::Scalar(255, 0, 0), 2, 8, 0);
    // }
    // cv::namedWindow("Feature Points in Previous Image");
    // cv::imshow("Feature Points in Previous Image", show_feature_pre);

    // cv::Mat show_feature_curerent;
    // rgb2.copyTo(show_feature_curerent);
    // for (int i = 0; i < kpts2.size(); i++)
    // {
    //     cv::circle(show_feature_curerent, kpts2[i].pt, 1, cv::Scalar(0, 0, 255), 2, 8, 0);
    // }
    // cv::namedWindow("Feature Points in Current Image");
    // cv::imshow("Feature Points in Current Image", show_feature_curerent);
    // cv::waitKey(0);

    std::vector<cv::DMatch> match;
    Ptr<cv::DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");
    matcher->match(desc1, desc2, match, cv::Mat());
    
    double minDis = 9999;
    for(size_t i=0; i<match.size(); i++)
    {
        if( match[i].distance < minDis )
        {
            minDis = match[i].distance;
            cout<<minDis<<" "<<std::endl;
        }
    }

    std::vector<cv::DMatch> goodmatches; //0->static; 1->random; 2->dynamic; 

    for(size_t i=0; i<match.size();i++)
    {
        if(match[i].distance < 4 * minDis)
        {
            goodmatches.push_back(match[i]);
        }
    }

    cv::Mat good_match_img;
    drawMatches(prevImg, kpts1, currentImg, kpts2, goodmatches, good_match_img);


    // cv::namedWindow("Original Previous Image");
    // cv::imshow("Original Previous Image", rgb1);


    // cv::namedWindow("Segmentation of Current Image");
    // cv::imshow("Segmentation of Current Image", rgb2);
    
    // cv::namedWindow("Grayscale of Previous Image");
    // cv::imshow("Grayscale of Previous Image", prevImg);
    // cv::waitKey(0);


     cv::namedWindow("Good Match");
    cv::imshow("Good Match", good_match_img);
    cv::waitKey(0);

}