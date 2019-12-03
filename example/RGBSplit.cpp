/*
Splict multi channel images into single chanels
*/
#include <iostream>
#include <opencv2/opencv.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <iostream>
#include <vector>

using namespace cv;
using namespace std;

int main(int argc, char **argv)
{
    String imageName("../data/pca_test.jpg"); // by default
    if (argc > 1)
    {
        imageName = argv[1];
    }
    Mat src = imread(imageName);

    // Check if image is loaded successfully
    if (!src.data || src.empty())
    {
        cout << "Problem loading image!!!" << endl;
        return EXIT_FAILURE;
    }

    imshow("src", src);

    cv::Mat chanels[3];
    split(src, chanels);

    imshow("B", chanels[0]);
    imshow("G", chanels[1]);
    imshow("R", chanels[2]);

    cv::Mat merged;
    merge(chanels, 3, merged);
    imshow("merged", merged);

    waitKey(0);
    return 0;
}