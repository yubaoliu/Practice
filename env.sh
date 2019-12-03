#OpenCV
export OpenCV_DIR="$HOME/software/opencv3.3.1/share/OpenCV"   

#CUDA
export CUDA_SDK_ROOT_DIR="/usr/local/cuda-10.0"
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"
export PATH="/usr/local/cuda-10.0/bin:${PATH}"
export CUDA_TOOKIT_ROOT_DIR="/usr/local/cuda-10.0"

# g2o
export g2o_DIR="$HOME/software/g2o/lib/cmake/g2o"

#PCL
export PCL_DIR="$HOME/software/install/pcl-1.8"


#glags
export gflags_DIR="~/software/install/gflags/lib/cmake/gflags"

#glog
export glog_DIR="~/software/install/glog"


#caffe
export CAFFE_ROOT="~/software/caffe"
export PYCAFFE_ROOT="$CAFFE_ROOT/python"
export PYTHONPATH="$PYCAFFE_ROOT:$PYTHONPATH"
export LD_LIBRARY_PATH="$CAFFE_ROOT/lib:$LD_LIBRARY_PATH"


