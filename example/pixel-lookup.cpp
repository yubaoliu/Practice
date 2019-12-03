#include "iostream"
#include "math.h"
#include "stdio.h"
#include "opencv2/opencv.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/core/core.hpp"
using namespace cv;
using namespace std;

//鼠标点击获取图像的像素值（包括彩色与黑白图像）//
void onMouse(int event, int x, int y, int flags, void *param)
{
    Mat *image = reinterpret_cast<Mat *>(param);
    switch (event)
    {
    case CV_EVENT_FLAG_LBUTTON:
        if (image->type() == CV_8UC1)
        { //gray image
            cout << "at(" << y << "," << x << ") value is: " << static_cast<int>(image->at<uchar>((Point(x, y)))) << endl;
        }
        else if (image->type() == CV_8UC3) //rgb(color) image
        {
            //cout<<"at("<<y<<","<<x<<") value is: "<<"( "<<(image->at<Vec3b>(y,x)[0])<<", "<<(image->at<Vec3b>(y,x)[1])<<", "<<(image->at<Vec3b>(y,x)[2])<<")"<<endl;
            cout << "at(" << y << "," << x << ") value is: "
                 << "( " << static_cast<int>(image->at<Vec3b>(y, x)[0]) << ", " << static_cast<int>(image->at<Vec3b>(y, x)[1]) << ", " << static_cast<int>(image->at<Vec3b>(y, x)[2]) << ")" << endl;
        }
        break;
    }
}

int main()
{
    // Mat image = imread("../data/walk/walk_rgb_2.png", CV_LOAD_IMAGE_COLOR);

    Mat image=imread("../data/walk/walk_rgb_2_label.png",CV_LOAD_IMAGE_GRAYSCALE);
    imshow("image", image);
    setMouseCallback("image", onMouse, reinterpret_cast<void *>(&image));

    waitKey(0);
    return 0;
}