

 ADD_EXECUTABLE( segnet_mono Segnet_mono.cpp    Segnet.cpp)

 INCLUDE_DIRECTORIES(${PROJECT_SOURCE_DIR}/Thirdparty/SegNet/SegNet/caffe-segnet/build/install/include )

TARGET_LINK_LIBRARIES( segnet_mono
    ${PROJECT_NAME}
    ${PROJECT_SOURCE_DIR}/Thirdparty/SegNet/SegNet/caffe-segnet/build/install/lib/libcaffe.so
    glog 
    # ${OpenCV_LIBS} 
    # ${PCL_LIBRARIES}
    )
    

ADD_EXECUTABLE( DSSLAM ds_slam.cpp)
TARGET_LINK_LIBRARIES( DSSLAM 
    ${PROJECT_NAME}
     )

ADD_EXECUTABLE( orb-seg orb-seg.cpp )
TARGET_LINK_LIBRARIES( orb-seg 
    ${PROJECT_NAME}
     )


ADD_EXECUTABLE( orb-kmeans orb-kmeans.cpp )
TARGET_LINK_LIBRARIES( orb-kmeans 
    ${PROJECT_NAME}
     )

ADD_EXECUTABLE( monitor-test monitor-test.cpp )
TARGET_LINK_LIBRARIES( monitor-test 
    ${PROJECT_NAME}
    )
