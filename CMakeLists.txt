CMAKE_MINIMUM_REQUIRED( VERSION 3.0 )
PROJECT( AISLAM )

if (POLICY CMP0074)
  cmake_policy(SET CMP0074 NEW)
endif()


LIST( APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules )

SET( CMAKE_CXX_COMPILER "g++") #设定编译器

# LIST( APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules )

# Customize variables
SET(ENV{OpenCV_DIR} "/home/yubao/data/software/OpenCV-3.3.1/share/OpenCV")
#SET(ENV{PCL_DIR} "/home/yubao/data/software/pcl-1.8/share/pcl-1.8")
SET(ENV{CUDA_SDK_ROOT_DIR} "/usr/local/cuda-10.0")  
SET(ENV{gflags_DIR}  "~/data/software/install/gflags/lib/cmake/gflags")
#SET(ENV{glog_DIR} "~/data/software/glog/lib/cmake/glog")
# SET(ENV{g2o_DIR} "/home/yubao/data/software/g2o/lib/cmake/g2o" )
#SET(ENV{octomap_DIR} "/home/yubao/data/software/octomap/share/octomap")
set(ENV{OpenPose_DIR} "/home/yubao/data/software/openpose/lib/OpenPose")
SET(ENV{G2O_ROOT} "/home/yubao/data/software/g2o") 

message($ENV{OpenCV_DIR})

# message(${GFLAGS_VERSION_STRING})


include(CheckCXXCompilerFlag)

CHECK_CXX_COMPILER_FLAG("-std=c++14" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
if(COMPILER_SUPPORTS_CXX11)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall  -std=c++11")
elseif(COMPILER_SUPPORTS_CXX0X)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
else()
        message(STATUS "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
endif()

# set(CMAKE_CXX_FLAGS_RELEASE "-O2 -DNDEBUG " ) # release包优化
set(CMAKE_CONFIGURATION_TYPES "Debug" CACHE STRING "" FORCE)


#设定可执行二进制文件的目录
SET( EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin) 
	
#设定存放编译出来的库文件的目录
SET( LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib) 
#并且把该目录设为连接目录
LINK_DIRECTORIES( ${PROJECT_SOURCE_DIR}/lib)

IF(CSPARSE_FOUND)
    message(${CSPARSE_INCLUDE_DIR})
endif()

# 增加PCL库的依赖
# FIND_PACKAGE( PCL 1.8 REQUIRED COMPONENTS common io visualization filters)
FIND_PACKAGE( PCL 1.7 REQUIRED)
list(REMOVE_ITEM PCL_LIBRARIES "vtkproj4") # use this in Ubuntu 16.04
# 添加头文件和库文件
ADD_DEFINITIONS( ${PCL_DEFINITIONS} )

# BOOST
find_package(Boost REQUIRED)
find_package(OpenCV  REQUIRED ) # the latest OpenCV
find_package(Eigen3 REQUIRED)
# find_package(glog REQUIRED)
find_package(CUDA REQUIRED)
find_package(gflags REQUIRED)
FIND_PACKAGE(G2O REQUIRED )
FIND_PACKAGE(CSparse REQUIRED )
FIND_PACKAGE(Cholmod)
FIND_PACKAGE(SuiteSparse)

# octomap
FIND_PACKAGE( octomap REQUIRED )

INCLUDE_DIRECTORIES( 
    ${PROJECT_SOURCE_DIR}/include
    /usr/local/include
    /usr/local/cuda/include
    /usr/include
    ${PCL_INCLUDE_DIRS}  
    ${OpenCV_INCLUDE_DIRS}
    ${Boost_INCLUDE_DIRS}
    ${G2O_INCLUDE_DIR}
    ${CSPARSE_INCLUDE_DIR}
    ${OCTOMAP_INCLUDE_DIRS}
    ${GFLAGS_INCLUDE_DIR} 
    # ${GLOG_INCLUDE_DIR}
)


aux_source_directory(src DIR_LIB_SRCS)
message(${DIR_LIB_SRCS})
message(${PCL_LIBRARY_DIRS})

add_library(${PROJECT_NAME} SHARED
#  ${DIR_LIB_SRCS}
    src/slamBase.cpp
    src/yb_algorithm.cpp
 )

 SET(G2O_LIBRARIES 
    ${G2O_STUFF_LIBRARY} 
    ${G2O_CORE_LIBRARY}
    ${G2O_SOLVER_CSPARSE}
    ${G2O_SOLVER_CHOLMOD}
    ${G2O_SOLVER_DENSE} 
    ${G2O_SOLVER_PCG} 
    ${G2O_SOLVER_SLAM2D_LINEAR}
    ${G2O_SOLVER_STRUCTURE_ONLY} 
    ${G2O_SOLVER_EIGEN} 
    ${G2O_SOLVER_CSPARSE}
    ${G2O_SOLVER_CSPARSE_EXTENSION}
    ${G2O_SOLVER_DENSE}
    ${G2O_TYPES_DATA}
    ${G2O_TYPES_ICP}
    ${G2O_TYPES_SBA}
    ${G2O_TYPES_SCLAM2D}
    ${G2O_TYPES_SIM3}
    ${G2O_TYPES_SLAM3D}
    ${CHOLMOD_LIBRARY}
 )


link_directories(${PCL_LIBRARY_DIRS})

TARGET_LINK_LIBRARIES( ${PROJECT_NAME}
    ${EIGEN3_LIBS}
    ${PCL_LIBRARIES}
    ${OpenCV_LIBS}
    ${Boost_LIBRARIES}
  #  -lboost_system
    ${G2O_LIBRARIES}
    ${CSPARSE_LIBRARY}
    ${OCTOMAP_LIBRARIES}
    ${GFLAGS_LIBRARIES} 
    # ${GLOG_LIBRARY} 
    glog
    )

message(${GFLAGS_INCLUDE_DIR} )
message(${GFLAGS_LIBRARIES})

#增加子文件夹，也就是进入源代码文件夹继续构建
ADD_SUBDIRECTORY( ${PROJECT_SOURCE_DIR}/src)
#ADD_SUBDIRECTORY( ${PROJECT_SOURCE_DIR}/test)
ADD_SUBDIRECTORY( ${PROJECT_SOURCE_DIR}/example)
