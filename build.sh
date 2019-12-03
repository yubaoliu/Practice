echo "Install necessary packages"


#YBlib
cd Thirdparty
git clone git@github.com:yubaoliu/YBLib.git

# Build SegNet
cd Thirdparty/SegNet/SegNet/caffe-segnet/
mkdir build
cd build
cmake ..
make
make install

