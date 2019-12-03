
#include <iostream>
#include <opencv2/core/core.hpp>
// #include <opencv2/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/ml/ml.hpp>

using namespace cv;
using namespace std;

int main(int argc, char *argv[])
{

	String imageName("../data/pca_test.jpg"); // by default

	if (argc > 1)
	{
		imageName = argv[1];
	}
	Mat img = imread(imageName);

	// Check if image is loaded successfully
	if (!img.data || img.empty())
	{
		cout << "Problem loading image!!!" << endl;
		return EXIT_FAILURE;
	}

	// Mat img = imread("../data/walk/walk_depth_1.png");
	namedWindow("Source Image", 0);
	imshow("Source Image", img);
	std::cout << "type" << img.type() << endl; //16->CV_8UC3
	//生成一维采样点,包括所有图像像素点,注意采样点格式为32bit浮点数。
	Mat samples(img.cols * img.rows, 1, CV_32FC3);
	//标记矩阵，32位整形
	Mat labels(img.cols * img.rows, 1, CV_32SC1);
	uchar *p;
	int i, j, k = 0;
	for (i = 0; i < img.rows; i++)
	{
		p = img.ptr<uchar>(i); //why uchar?
		for (j = 0; j < img.cols; j++)
		{
			samples.at<Vec3f>(k, 0)[0] = float(p[j * 3]);
			samples.at<Vec3f>(k, 0)[1] = float(p[j * 3 + 1]);
			samples.at<Vec3f>(k, 0)[2] = float(p[j * 3 + 2]);
			k++;
		}
	}

	int clusterCount = 6;
	Mat centers(clusterCount, 1, samples.type());
	kmeans(samples, clusterCount, labels,
		   TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 10, 1.0),
		   3, KMEANS_PP_CENTERS, centers);
	//我们已知有n个聚类，用不同的灰度层表示。
	Mat img1(img.rows, img.cols, CV_8UC1);
	float step = 255 / (clusterCount - 1);
	k = 0;
	for (i = 0; i < img1.rows; i++)
	{
		p = img1.ptr<uchar>(i);
		for (j = 0; j < img1.cols; j++)
		{
			int tt = labels.at<int>(k, 0);
			k++;
			p[j] = 255 - tt * step;
		}
	}

	namedWindow("K-Means分割效果", 0);
	imshow("K-Means分割效果", img1);
	imwrite("kmeans.png", img1);
	waitKey(0);
	return 0;
}
