// The "Square Detector" program.
// It loads several images sequentially and tries to find squares in
// each image

#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"

#include <iostream>
#include <math.h>
#include <string.h>

using namespace cv;
using namespace std;

// helper function:
// finds a cosine of angle between vectors
// from pt0->pt1 and from pt0->pt2
static double angle(Point pt1, Point pt2, Point pt0)
{
    double dx1 = pt1.x - pt0.x;
    double dy1 = pt1.y - pt0.y;
    double dx2 = pt2.x - pt0.x;
    double dy2 = pt2.y - pt0.y;
    return (dx1 * dx2 + dy1 * dy2) / sqrt((dx1 * dx1 + dy1 * dy1) * (dx2 * dx2 + dy2 * dy2) + 1e-10);
}

Mat src, src_gray;
Mat dst, detected_edges;
int edgeThresh = 1;
int lowThreshold = 0;
int const max_lowThreshold = 100;
int ratio = 3;
int kernel_size = 3;
const char *window_name = "Edge Map";

cv::Mat CannyThreshold(int thresh)
{
    blur(src_gray, detected_edges, Size(3, 3));
    Canny(detected_edges, detected_edges, thresh, thresh * ratio, kernel_size);
    // cv::Mat gray;
    // dilate(detected_edges, gray, Mat(), Point(-1, -1));

    dst = Scalar::all(0);
    src.copyTo(dst, detected_edges);
    imshow(window_name, dst);
    return dst;
}

int main(int argc, char **argv)
{
    cv::Mat depth;
    src = imread(argv[1], IMREAD_COLOR); // Load an image
    if (src.empty())
    {
        return -1;
    }
    cout << "Source image type: " << src.type() << endl;

    if (argc == 3)
    {
        depth = cv::imread(argv[2], -1);
        cout << "depth.type: " << depth.type() << endl;

        //Calculate the min and max depth
        double minDepth = 999.99;
        double maxDepth;
        // cv::Point min_loc, max_loc;
        // cv::minMaxLoc(depth, &minDepth, &maxDepth, &min_loc, &max_loc);
        cout << "depth: ";
        for (int i = 0; i < depth.rows; i++)
        {
            for (int j = 0; j < depth.cols; j++)
            {
                float d = depth.ptr<uchar>(i)[j];
                if (d < 0.5)
                {
                    continue;
                }
                if (d < minDepth)
                {
                    minDepth = d;
                }
                if (d > maxDepth)
                {
                    maxDepth = d;
                }
                // cout << d << " ";
            }
        }
        cout << "minDepth: " << minDepth << endl;
        cout << "maxDepth: " << maxDepth << endl;

        double averageDepth = (maxDepth - minDepth) / 2;

        for (int i = 0; i < src.rows; i++)
        {
            for (int j = 0; j < src.cols; j++)
            {
                float d = depth.ptr<uchar>(i)[j];
                if (d > averageDepth)
                {
                    src.at<cv::Vec3b>(i, j)[0] = 0;
                    src.at<cv::Vec3b>(i, j)[1] = 0;
                    src.at<cv::Vec3b>(i, j)[2] = 0;
                }
            }
        }
        cv::imshow("Source image after remove remost pixel", src);
    }

    cout << endl;

    dst.create(src.size(), src.type());
    cvtColor(src, src_gray, COLOR_BGR2GRAY);
    // namedWindow(window_name, WINDOW_AUTOSIZE);

    cv::Mat result;
    vector<vector<Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    result = CannyThreshold(100);
    cout << result.type() << endl;

    cv::Mat input;
    cvtColor(result, input, COLOR_BGR2GRAY);

    findContours(input, contours, RETR_LIST, CHAIN_APPROX_NONE);

    double maxArea = 0.0;
    for (size_t i = 0; i < contours.size(); ++i)
    {
        double area = cv::contourArea(contours[i]);
        if (maxArea < area)
        {
            maxArea = area;
        }
    }

    cout << "maxArea: " << maxArea << endl;

    for (size_t i = 0; i < contours.size(); ++i)
    {
        double area = cv::contourArea(contours[i]);

        if (area < maxArea / 10.0 || area < 5)
        {
            // cout << "area: " << area << endl;
            continue;
        }

        drawContours(src, contours, static_cast<int>(i), cv::Scalar(0, 0, 255), 2, 8, hierarchy, 0);
    }

    imshow("countours", src);
    waitKey(0);
    return 0;

    // namedWindow(wndname, 1);
    // vector<vector<Point>> squares;

    // for (int i = 0; names[i] != 0; i++)
    // {
    //     Mat image = imread(names[i], 1);
    //     if (image.empty())
    //     {
    //         cout << "Couldn't load " << names[i] << endl;
    //         continue;
    //     }

    //     findSquares(image, squares);
    //     drawSquares(image, squares);

    //     char c = (char)waitKey();
    //     if (c == 27)
    //         break;
    // }

    return 0;
}
