#ifndef _COMMON_H_
#define _COMMON_H_

//C++ std
#include <iostream>
#include <algorithm>
#include <vector>
#include <fstream>
#include <mutex>
#include <thread>

//BOOST
#include <boost/array.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/scoped_ptr.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/algorithm/string/trim.hpp>

//Eigen
#include <Eigen/Dense>

//OpenCV
#include <opencv2/core/eigen.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/video/tracking.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/ml/ml.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d.hpp> //ORB

// PCL
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/common/transforms.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/filters/voxel_grid.h>
// #include <pcl_conversions/pcl_conversions.h>

//log
#include <glog/logging.h>

#endif