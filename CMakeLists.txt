cmake_minimum_required(VERSION 2.8)
project(cuda-memory-access)

find_package(CUDA REQUIRED)
find_package( OpenCV REQUIRED )
find_package(benchmark REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE RelWithDebInfo)
set(CMAKE_CUDA_FLAGS "-arch=sm_53")

include_directories( ${OpenCV_INCLUDE_DIRS} ${CUDA_INCLUDE_DIRS})

add_executable( bench bench.cpp )
target_link_libraries( bench ${OpenCV_LIBS} benchmark::benchmark ${CUDA_LIBRARIES} )
