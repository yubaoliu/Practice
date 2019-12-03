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
// using namespace YB_SLAM;
// struct Obj_Info_s
// {
//     int feature_num;
//     int move_num;
// };
struct Object_Label_s
{
    int label;
    char class_name[20];
};
const int MAX_FEATURES = 1000;
// const float GOOD_MATCH_PERCENT = 0.15f;
void priori_object_info(std::map<int, string> &static_object_m, std::map<int, string> &dynamic_object_m)
{
    static_object_m.insert({3, "floor"});
    static_object_m.insert({5, "ceiling"});
    static_object_m.insert({8, "window"});
    static_object_m.insert({19, "chair"});
    static_object_m.insert({33, "desk"});
    static_object_m.insert({74, "computer"});
    static_object_m.insert({75, "chair"});
    static_object_m.insert({100, "card"});
    static_object_m.insert({143, "monitor"});
    dynamic_object_m.insert({12, "person"});
}
bool monitor_exists_f = false;

int main(int argc, char **argv)
{

    //person or other dynamic objects in the monitor
    cv::Mat rgb1 = cv::imread("../data/lovelive/frame0059.jpg");
    cv::Mat label1 = cv::imread("../data/lovelive/lovelive_0059_label.png", cv::IMREAD_GRAYSCALE);

    cv::Mat rgb2 = cv::imread("../data/lovelive/frame0060.jpg");
    cv::Mat label2 = cv::imread("../data/lovelive/lovelive_0060_label.png", cv::IMREAD_GRAYSCALE);

    cv::resize(rgb1, rgb1, cv::Size(640, 480));
    cv::resize(label1, label1, cv::Size(640, 480));
    // cv::resize(depth1, depth1, cv::Size(640, 480));

    cv::resize(rgb2, rgb2, cv::Size(640, 480));
    cv::resize(label2, label2, cv::Size(640, 480));
    // cv::resize(depth2, depth2, cv::Size(640, 480));

    cv::Mat prevImg;
    cv::Mat currentImg;
    cv::cvtColor(rgb1, prevImg, cv::COLOR_BGR2GRAY);
    cv::cvtColor(rgb2, currentImg, cv::COLOR_RGB2GRAY);

    if (prevImg.empty() || currentImg.empty() || prevImg.channels() != 1 || currentImg.channels() != 1)
    {
        cout << "Input Image is nullptr or the image channels is not gray!" << endl;
        system("pause");
    }


    /* Feature extraction*/
    Ptr<cv::Feature2D> orb = ORB::create(MAX_FEATURES, 1.2, 8, 31, 0, 2, 0, 31, 20);
    vector<cv::KeyPoint> kpts1, kpts2;
    cv::Mat desc1, desc2;
    // 特征点检测算法...
    orb->detect(prevImg, kpts1);
    orb->detect(currentImg, kpts2);

    /*Feature Cluster*/
    std::map<int, string> priori_static_m, priori_dynamic_m;
    priori_object_info(priori_static_m, priori_dynamic_m);

    /*Object Cluster*/
    int obj_num_img_current = 0;
    std::vector<int> obj_label;
    for (int r = 0; r < label2.rows; r++)
    {
        for (int c = 0; c < label2.rows; c++)
        {
            int l = (int)label2.at<uchar>(r, c);
            std::vector<int>::iterator it;
            it = find(obj_label.begin(), obj_label.end(), l);
            if (l != 0 && it == obj_label.end())
            {
                obj_label.push_back(l);
            }
        }
    }

    std::cout << "Object label: " << std::endl;
    std::sort(obj_label.begin(), obj_label.end());
    for (size_t i = 0; i < obj_label.size(); i++)
    {
        std::cout << obj_label.at(i) << ' ';
        if (priori_static_m.find(obj_label.at(i))->second == string("monitor"))
        {
            monitor_exists_f = true;
        }
    }
    std::cout << std::endl;

    obj_num_img_current = obj_label.size();
    std::cout << "Object totlal number in previous image:" << obj_num_img_current << std::endl;

    std::vector<cv::KeyPoint> previous_feature[3]; //0->static, 1->random, 2->dynamic
    // std::vector<cv::KeyPoint> previous_static, previous_dynamic, previous_random;

    for (uint i = 0; i < kpts1.size(); i++)
    {
        kpts1[i].class_id = (int)label1.at<uchar>(kpts1[i].pt);
        if (priori_static_m.find(kpts1[i].class_id) != priori_static_m.end())
        {
            previous_feature[0].push_back(kpts1[i]);
        }
        else if (priori_dynamic_m.find(kpts1[i].class_id) != priori_dynamic_m.end()) //dynamic, 12-> person
        {
            previous_feature[2].push_back(kpts1[i]);
        }
        else
        { //random
            previous_feature[1].push_back(kpts1[i]);
        }
    }

    std::vector<cv::KeyPoint> current_feature[3];

    cv::Mat canny_output;
    vector<vector<cv::Point> > contours;
    vector<Vec4i> hierarchy;

    if (monitor_exists_f)
    {
        cv::Canny(label2, canny_output, 100, 200);
        cv::findContours(canny_output, contours, hierarchy, CV_RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

        Mat drawing = Mat::zeros( canny_output.size(), CV_8UC3 );
        RNG rng(12345);
        for( size_t i = 0; i< contours.size(); i++ )
        {
            Scalar color = Scalar( rng.uniform(0, 256), rng.uniform(0,256), rng.uniform(0,256) );
            drawContours( drawing, contours, (int)i, color, 2, LINE_8, hierarchy, 0 );
        }

        cv::imshow( "Contours", drawing );
        cv::imshow("canny_output", canny_output);
        cv::waitKey(0); 
    }

    for (uint i = 0; i < kpts2.size(); i++)
    {
        kpts2[i].class_id = (int)label2.at<uchar>(kpts2[i].pt);
        // kpts2[i].class_id = (int)label2.at<uchar>((int)kpts2[i].pt.y, (int)kpts2[i].pt.x);
        if (priori_static_m.find(kpts2[i].class_id) != priori_static_m.end())
        {
            // if(priori_static_m.find(kpts1[i].class_id)->second  == "monitor")
            // {
            //     int label = priori_static_m.find(kpts1[i].class_id)->first;
            //     int dx[9] = {-1, 0, 1, -1, 0, 1, -1, 0, 1};
            //     int dy[9] = {-1, -1, -1, 0, 0, 0, 1, 1, 1};
            //     float x = kpts1[i].pt.x;
            //     float y = kpts1[i].pt.y;
            //     int k = 0;
            //     while (label2.at<uchar>(x+dx[k++], y+dy[k++]) == label);
            //     if (k<8){
            //         current_feature[0].push_back(kpts2[i]); //edge point of monitor
            //     }else{
            //         current_feature[2].push_back(kpts2[i]);
            //     }
            // }else{
            current_feature[0].push_back(kpts2[i]);
            // }
        }
        else if (priori_dynamic_m.find(kpts2[i].class_id) != priori_dynamic_m.end()) //dynamic, 12-> person
        {
            current_feature[2].push_back(kpts2[i]);
        }
        else
        { //random
            current_feature[1].push_back(kpts2[i]);
        }
    }

    // cv::Mat show_current_feature[3];
    // for(size_t i=0; i<3; i++)
    // {
    //     rgb2.copyTo(show_current_feature[i]);
    //     for (size_t j = 0; j < current_feature[i].size(); j++)
    //     {
    //         cv::circle(show_current_feature[i], current_feature[i][j].pt, 1, cv::Scalar(0, 0, 255), 2, 8, 0);
    //     }
    //     char buf[50];
    //     sprintf(buf, "Feature Points in Current Image [%u]", i);
    //     cv::imshow(buf, show_current_feature[i]);
    //     cv::waitKey(0);
    // }

    cv::Mat previous_desp[3], current_desp[3];
    std::vector<cv::DMatch> match[3];
    Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create("BruteForce-Hamming");
    for (size_t i = 0; i < 3; i++)
    {
        orb->compute(prevImg, previous_feature[i], previous_desp[i]);
        orb->compute(currentImg, current_feature[i], current_desp[i]);
        matcher->match(previous_desp[i], current_desp[i], match[i], cv::Mat());
    }

    //Calculate minimal distance
    double minDis = 9999;
    for (size_t i = 0; i < 3; i++)
    {
        for (size_t j = 0; j < match[i].size(); j++)
        {
            if (match[i][j].distance > 5 && match[i][j].distance < minDis) //2.15614e-25 >0, 5 is randomly set
            {
                minDis = match[i][j].distance;
                // cout<<minDis<<" "<<std::endl; //used for debug the minimal distance
            }
        }
    }
    cout << "Minimal distance: " << minDis << std::endl;

    /*Good Match*/
    std::vector<cv::DMatch> goodmatches[3]; //0->static; 1->random; 2->dynamic;
    for (size_t i = 0; i < 3; i++)
    {
        for (size_t j = 0; j < match[i].size(); j++)
        {
            if (match[i][j].distance < 4 * minDis)
            {
                goodmatches[i].push_back(match[i][j]);
            }
        }
    }

    Mat good_match_img[3];
    for (size_t i = 0; i < 3; i++)
    {
        drawMatches(prevImg, previous_feature[i], currentImg, current_feature[i], goodmatches[i], good_match_img[i]);
    }

    for (size_t i = 0; i < 3; i++)
    {
        if (good_match_img[i].dims > 0)
        {
            char buf[50];
            sprintf(buf, "Good Matches [%d]", (int)i);
            cv::imshow(buf, good_match_img[i]);
        }
    }

    monitor_exists_f = false;

    cv::waitKey(0);

    // //设置角点检测参数
    // std::vector<cv::Point2f> prePts, nextPts, prePts_tmp, nextPts_tmp, nextPtsInlier, prePtsInlier, nextPtsOutlier, prePtsOutlier;
    // int max_corners = 1000;
    // double quality_level = 0.01;
    // double min_distance = 3.0;
    // int block_size = 3;
   
    // //Remove Outlier
    // // Then if the matched pair is too close to the edge of the image or the pixel difference of the 3×3 image block at the center of the matched pair is too large, the matched pair will be discarded.
    // int limit_edge_corner = 5;
    // double limit_of_check = 2120;
    // for (int i = 0; i < status.size(); i++)
    // {
    //     if (status[i] != 0)
    //     {
    //         int dx[10] = {-1, 0, 1, -1, 0, 1, -1, 0, 1};
    //         int dy[10] = {-1, -1, -1, 0, 0, 0, 1, 1, 1};
    //         int x1 = prePts[i].x, y1 = prePts[i].y;
    //         int x2 = nextPts[i].x, y2 = nextPts[i].y;
    //         if ((x1 < limit_edge_corner || x1 >= nextImg.cols - limit_edge_corner || x2 < limit_edge_corner || x2 >= nextImg.cols - limit_edge_corner || y1 < limit_edge_corner || y1 >= nextImg.rows - limit_edge_corner || y2 < limit_edge_corner || y2 >= nextImg.rows - limit_edge_corner))
    //         {
    //             status[i] = 0;
    //             continue;
    //         }
    //         double sum_check = 0;
    //         for (int j = 0; j < 9; j++)
    //             sum_check += abs(prevImg.at<uchar>(y1 + dy[j], x1 + dx[j]) - nextImg.at<uchar>(y2 + dy[j], x2 + dx[j]));
    //         if (sum_check > limit_of_check)
    //             status[i] = 0;
    //         if (status[i])
    //         {
    //             prePtsInlier.push_back(prePts[i]);
    //             nextPtsInlier.push_back(nextPts[i]);
    //         }
    //     }
    // }

    // //The third step is to find fundamental matrix by using RANSAC with the most inliers. Then calculate the epipolar line in the current frame by using the fundamental matrix. Finally, determine whether the distance from a matched point to its corresponding epipolar line is less than a certain threshold. If the distance is greater than the threshold, then the matched point will be determined to be moving.
    // // F-Matrix
    // cv::Mat mask = cv::Mat(cv::Size(1, 300), CV_8UC1);
    // cv::Mat F = cv::findFundamentalMat(prePtsInlier, nextPtsInlier, mask, cv::FM_RANSAC, 0.1, 0.99);
    // for (int i = 0; i < mask.rows; i++)
    // {
    //     if (mask.at<uchar>(i, 0) == 0)
    //         ;
    //     else
    //     {
    //         // Circle(pre_frame, F_prepoint[i], 6, Scalar(255, 255, 0), 3);
    //         double A = F.at<double>(0, 0) * prePtsInlier[i].x + F.at<double>(0, 1) * prePtsInlier[i].y + F.at<double>(0, 2);
    //         double B = F.at<double>(1, 0) * prePtsInlier[i].x + F.at<double>(1, 1) * prePtsInlier[i].y + F.at<double>(1, 2);
    //         double C = F.at<double>(2, 0) * prePtsInlier[i].x + F.at<double>(2, 1) * prePtsInlier[i].y + F.at<double>(2, 2);
    //         double dd = fabs(A * nextPtsInlier[i].x + B * nextPtsInlier[i].y + C) / sqrt(A * A + B * B); //Epipolar constraints
    //         if (dd <= 0.1)
    //         {
    //             prePts_tmp.push_back(prePtsInlier[i]);
    //             nextPts_tmp.push_back(nextPtsInlier[i]);
    //         }
    //     }
    // }
    // prePtsInlier = prePts_tmp;
    // nextPtsInlier = nextPts_tmp;

    // //judge outliers
    // for (int i = 0; i < prePts.size(); i++)
    // {
    //     if (status[i] != 0)
    //     {
    //         double A = F.at<double>(0, 0) * prePts[i].x + F.at<double>(0, 1) * prePts[i].y + F.at<double>(0, 2);
    //         double B = F.at<double>(1, 0) * prePts[i].x + F.at<double>(1, 1) * prePts[i].y + F.at<double>(1, 2);
    //         double C = F.at<double>(2, 0) * prePts[i].x + F.at<double>(2, 1) * prePts[i].y + F.at<double>(2, 2);
    //         double dd = fabs(A * nextPts[i].x + B * nextPts[i].y + C) / sqrt(A * A + B * B);

    //         // Judge outliers
    //         double limit_dis_epi = 1;
    //         if (dd <= limit_dis_epi)
    //             continue;
    //         nextPtsOutlier.push_back(nextPts[i]);
    //     }
    // }

    // //visualize
    // for (int i = 0; i < nextPtsOutlier.size(); i++)
    // {
    //     cv::circle(rgb2, nextPtsOutlier[i], 1, cv::Scalar(0, 0, 255), 2, 8, 0);
    // }

    // // cv::imshow("OutlierPoints", rgb2);
    // // cv::waitKey(0);

    // //visualize inliers
    // for (int i = 0; i < nextPtsInlier.size(); i++)
    // {
    //     cv::circle(rgb2, nextPtsInlier[i], 1, cv::Scalar(0, 0, 255), 2, 8, 0);
    // }

    // cv::imshow("nextPtsInlier", rgb2);
    // cv::waitKey(0);

    return 0;
}