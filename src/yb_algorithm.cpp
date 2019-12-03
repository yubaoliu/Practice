#include "yb_algorithm.h"

using namespace std;

namespace  RE_SLAM
{
std::vector<int> unique(const cv::Mat &input, bool sort = false)
{
    if (input.channels() > 1 || input.type() != CV_8UC1)
    {
        std::cerr << "unique !!! Only works with CV_(UC1) 1-channel Mat" << std::endl;
        return std::vector<int>();
    }

    std::vector<int> out;
    for (int y = 0; y < input.rows; ++y)
    {
        const int *row_ptr = input.ptr<int>(y);
        for (int x = 0; x < input.cols; ++x)
        {
            int value = row_ptr[x];

            if (std::find(out.begin(), out.end(), value) == out.end())
                out.push_back(value);
        }
    }

    if (sort)
        std::sort(out.begin(), out.end());

    return out;
}

} // namespace  YB_SLAM

