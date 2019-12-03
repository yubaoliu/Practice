//./connectedComponents ../data/connect_test.png
#include <algorithm>
#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
int main(int argc, char *argv[])
{
    cv::Mat src_img, img_bool, labels, stats, centroids, img_color, img_gray;

    if ((src_img = cv::imread(argv[1], cv::IMREAD_GRAYSCALE)).empty())
    {
        cout << "load image error！";
        return -1;
    }

    cv::threshold(src_img, img_bool, 0, 255, cv::THRESH_OTSU);
    //连通域计算
    int nccomps = cv::connectedComponentsWithStats(
        img_bool, //二值图像
        labels,   //和原图一样大的标记图
        stats,    //nccomps×5的矩阵 表示每个连通区域的外接矩形和面积（pixel）
        centroids //nccomps×2的矩阵 表示每个连通区域的质心
    );

    //显示原图统计结果
    char title[1024];
    sprintf(title, "原图中连通区域数：%d\n", nccomps);
    cv::String num_connect(title);
    cv::imshow(num_connect, img_bool);

    //去除过小区域，初始化颜色表
    vector<cv::Vec3b> colors(nccomps);
    colors[0] = cv::Vec3b(0, 0, 0); // background pixels remain black.
    for (int i = 1; i < nccomps; i++)
    {
        colors[i] = cv::Vec3b(rand() % 256, rand() % 256, rand() % 256);
        //去除面积小于100的连通域
        if (stats.at<int>(i, cv::CC_STAT_AREA) < 100)
            colors[i] = cv::Vec3b(0, 0, 0); // small regions are painted with black too.
    }
    //按照label值，对不同的连通域进行着色
    img_color = cv::Mat::zeros(src_img.size(), CV_8UC3);
    for (int y = 0; y < img_color.rows; y++)
        for (int x = 0; x < img_color.cols; x++)
        {
            int label = labels.at<int>(y, x);
            CV_Assert(0 <= label && label <= nccomps);
            img_color.at<cv::Vec3b>(y, x) = colors[label];
        }

    //统计降噪后的连通区域
    cv::cvtColor(img_color, img_gray, cv::COLOR_BGR2GRAY);
    cv::threshold(img_gray, img_gray, 1, 255, cv::THRESH_BINARY);
    nccomps = cv::connectedComponentsWithStats(img_gray, labels, stats, centroids);
    sprintf(title, "过滤小目标后的连通区域数量：%d\n", nccomps);
    num_connect = title;
    cv::imshow(num_connect, img_color);
    cv::imshow("Labeled map", img_color);
    cv::waitKey();
    return 0;
}