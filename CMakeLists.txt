# Set  the minimum  required version  of cmake  for a  project.
cmake_minimum_required(VERSION 2.6)

# C++11
SET(CMAKE_CXX_FLAGS "-std=c++0x")

# VTK directory
set(VTK_DIR "/home/sungho/1_Data/VTK/VTK-8.2.0/build")

# library package
FIND_PACKAGE(k4a REQUIRED)
FIND_PACKAGE(k4abt REQUIRED)
FIND_PACKAGE(OpenCV REQUIRED)
find_package(VTK REQUIRED)
find_package(Geant4 REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})
link_directories(${OpenCV_LIBRARY_DIRS})
add_definitions(${OpenCV_DEFINITIONS})

#include directory
include_directories(${PROJECT_SOURCE_DIR}/include)
include(${VTK_USE_FILE})
include(${Geant4_USE_FILE})

# locate source files
file(GLOB headers ${PROJECT_SOURCE_DIR}/include/*)

# Add an executable to the project using the specified source files.

add_executable(C-armTracker main.cpp ${headers})
# link_directories(/home/hurel/Azure-Kinect-Sensor-SDK/build/lib)
target_link_libraries( C-armTracker k4a::k4a ${OpenCV_LIBS} ${VTK_LIBRARIES} ${Geant4_LIBRARIES})
