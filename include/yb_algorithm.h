#ifndef _ALGORITHM_H_
#define _ALGORITHM_H_
#include <opencv2/core/core.hpp>
#include <algorithm>
#include <vector>
#include <iostream>
#include <iosfwd>

namespace RE_SLAM
{
std::vector<int> unique(const cv::Mat &input, bool sort);

}
#endif