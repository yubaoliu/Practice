#include <iostream>
#include <algorithm>  
#include <vector>
#include <boost/array.hpp>
#include <boost/algorithm/string.hpp>
#include <Eigen/Dense>
#include <opencv2/core/eigen.hpp>
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

// void NCCCheck()
// {
//     // Then if the matched pair is too close to the edge of the image or the pixel difference of the 3×3 image block at the center of the matched pair is too large, the matched pair will be discarded.
//     int limit_edge_corner = 5;
//     double limit_of_check = 2120;
//     for (int i = 0; i < status.size(); i++)
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

int main(int argc, char **argv)
{
    cv::Mat rgb1 = cv::imread("../data/walk/walk_rgb_1.png");
    cv::Mat label1 = cv::imread("../data/walk/walk_rgb_1_label.png", cv::IMREAD_GRAYSCALE);
    cv::Mat depth1 = cv::imread("../data/walk/walk_depth_1.png");

    cv::Mat rgb2 = cv::imread("../data/walk/walk_rgb_2.png");
    cv::Mat depth2 = cv::imread("../data/walk/walk_depth_2.png");
    cv::Mat label2 = cv::imread("../data/walk/walk_rgb_2_label.png", cv::IMREAD_GRAYSCALE);

    cv::resize(rgb1, rgb1, cv::Size(640, 480));
    cv::resize(label1, label1,  cv::Size(640, 480));
    cv::resize(depth1, depth1, cv::Size(640, 480));

    cv::resize(rgb2, rgb2, cv::Size(640, 480));
    cv::resize(label2, label2, cv::Size(640, 480));
    cv::resize(depth2, depth2, cv::Size(640, 480));

    cv::Mat prevImg;
    cv::Mat currentImg;
    cv::cvtColor(rgb1, prevImg, cv::COLOR_BGR2GRAY);
    cv::cvtColor(rgb2, currentImg, cv::COLOR_RGB2GRAY);


    if(prevImg.empty() || currentImg.empty() || prevImg.channels() != 1 || currentImg.channels() != 1)
    {
        cout<<"Input Image is nullptr or the image channels is not gray!" << endl;
        system("pause");
    }

    /* Feature extraction*/
    Ptr<cv::Feature2D> orb = ORB::create(MAX_FEATURES, 1.2, 8, 31, 0, 2, 0, 31, 20);
    vector<cv::KeyPoint> kpts1, kpts2;
    cv::Mat desc1, desc2;
    // 特征点检测算法...
    orb->detect(prevImg, kpts1);
    orb->detect(currentImg, kpts2);

    /*Object Cluster*/
    int obj_num_img_current =  0;
    std::vector<int> obj_label;
    for(int r=0; r < label2.rows; r++)
    {
        for(int c=0; c<label2.rows; c++)
        {
            int l = (int)label2.at<uchar>(r, c);
            std::vector<int>::iterator it = find(obj_label.begin(), obj_label.end(), l);
            if( l != 0 && it == obj_label.end())
            {
                obj_label.push_back(l);
            }
        }
    }

    std::cout<<"Object label: "<<std::endl;
    std::sort(obj_label.begin(), obj_label.end());
    for (size_t i=0; i<obj_label.size(); i++){
        std::cout<<obj_label.at(i)<<' ';
    }
    std::cout<<std::endl;


    obj_num_img_current = obj_label.size();
    std::cout<<"Object totlal number in previous image:"<<obj_num_img_current<<std::endl;
   
    /*Feature Cluster*/
    std::map<int, string> priori_static_m, priori_dynamic_m;
    priori_object_info(priori_static_m, priori_dynamic_m);


    std::vector<cv::KeyPoint> previous_feature[3]; //0->static, 1->random, 2->dynamic
    // std::vector<cv::KeyPoint> previous_static, previous_dynamic, previous_random;

    for(uint i=0; i<kpts1.size(); i++)
    {
        kpts1[i].class_id = (int)label1.at<uchar>(kpts1[i].pt);
        if( priori_static_m.find(kpts1[i].class_id) != priori_static_m.end())
        {
           previous_feature[0].push_back(kpts1[i]);
        }else if( priori_dynamic_m.find(kpts1[i].class_id) != priori_dynamic_m.end()) //dynamic, 12-> person
        {
            previous_feature[2].push_back(kpts1[i]); 
        }else{ //random
            previous_feature[1].push_back(kpts1[i]);
        }
    }


    std::vector<cv::KeyPoint> current_feature[3];
    // std::vector<cv::KeyPoint> current_static, current_dynamic, current_random;
    for(uint i=0; i<kpts2.size(); i++)
    {
        kpts2[i].class_id = (int)label2.at<uchar>(kpts2[i].pt);
        // kpts2[i].class_id = (int)label2.at<uchar>((int)kpts2[i].pt.y, (int)kpts2[i].pt.x);
        if( priori_static_m.find(kpts2[i].class_id) != priori_static_m.end())
        {
            current_feature[0].push_back(kpts2[i]); 
        }else if( priori_dynamic_m.find(kpts2[i].class_id) != priori_dynamic_m.end()) //dynamic, 12-> person
        {
            current_feature[2].push_back(kpts2[i]); 
        }else{ //random
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

    
 

    //Geometry Moving Check
    // cv::Mat mask;
    std::vector<Point2f> points_pre, points_cur;
    size_t size = (previous_feature[0].size()>current_feature[0].size())?current_feature[0].size(): previous_feature[0].size();
    for(size_t i=0; i<size; i++)
    {
        points_pre.push_back(previous_feature[0][i].pt);
        points_cur.push_back(current_feature[0][i].pt);
    }

    // auto ip = std::unique(points_pre.begin(), points_pre.end());
    // previous_feature[0].resize(std::distance(points_pre.begin(), ip));

    // ip = std::unique(points_cur.begin(), points_cur.end());
    // current_feature[0].resize(std::distance(points_cur.begin(), ip));

    // size = (points_pre.size()>points_cur.size())?points_cur.size(): points_pre.size();
    
    cv::Mat mask(cv::Size(1, size), CV_8UC1);
    cout<<"Fundamental Matrix:"<<std::endl;
    cv::Mat F = cv::findFundamentalMat(points_pre, points_cur, mask, cv::FM_RANSAC, 0.1, 0.99);
    for(int i=0;i<F.rows;i++)
        for(int j=0; j<F.cols;j++)
            cout<<(int)F.at<uchar>(i,j)<<" ";
    std::cout<<endl;
    std::cout<<"Error residual:"<<endl;
    for(int i=0; i<mask.rows; i++)
    {
        if(mask.at<uchar>(i, 0) == 0) //outliers
        {
            for(auto it = current_feature[0].begin(); it != current_feature[0].end(); it++)
            {
                if(((int)it->pt.x == (int)points_cur[i].x ) && ((int)it->pt.y == (int)points_cur[0].y ))
                {
                    current_feature[0].erase(it);
                    current_feature[2].push_back(*it);
                    // std::cout<<it->pt<<" ";
                }
            }
        }else{  //inliner
            Eigen::Vector3f p1(points_pre[i].x, points_pre[i].y, 1);
            Eigen::Vector3f p2(points_cur[i].x, points_cur[i].y, 1);
            Eigen::Matrix3f FM;
            cv::cv2eigen(F, FM);
            int error = p2.transpose()*FM*p1;
            if(error>1)
            {
                std::cout<<error<<" ";
            }
        }
    }

cv::Mat previous_desp[3], current_desp[3];
    std::vector<cv::DMatch> match[3];
    Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create("BruteForce-Hamming");
    for(size_t i=0; i<3; i++)
    {
        orb->compute(prevImg, previous_feature[i], previous_desp[i]);
        orb->compute(currentImg, current_feature[i], current_desp[i]);
        matcher->match(previous_desp[i], current_desp[i], match[i], cv::Mat());
    }
  
    //Calculate minimal distance 
    double minDis = 9999;
    for(size_t i=0; i<3; i++)
    {
        for(size_t j=0; j<match[i].size(); j++)
        {
            if( match[i][j].distance > 5 && match[i][j].distance < minDis ) //2.15614e-25 >0, 5 is randomly set
            {
                minDis = match[i][j].distance;
                // cout<<minDis<<" "<<std::endl; //used for debug the minimal distance
            }
        }
    }
    cout<<"Minimal distance: "<<minDis<<std::endl;

    /*Good Match*/
    std::vector<cv::DMatch> goodmatches[3]; //0->static; 1->random; 2->dynamic; 
    for(size_t i=0; i<3; i++)
    {
        for(size_t j=0; j<match[i].size();j++)
        {
            if(match[i][j].distance < 4 * minDis)
            {
                goodmatches[i].push_back(match[i][j]);
            }
        }
    }
    
    Mat good_match_img[3];
    for(size_t i=0;i<3;i++)
    {
        drawMatches(prevImg, previous_feature[i], currentImg, current_feature[i], goodmatches[i], good_match_img[i]);
    }

    for(size_t i=0;i<3;i++)
    {
        if(good_match_img[i].data)
        {
            char buf[50];
            sprintf(buf,"Good Matches [%d]", (int)i );
            cv::imshow(buf, good_match_img[i]);
        }
    }

    std::cout<<"END"<<std::endl;
    cv::waitKey(0);

    return 0;
}