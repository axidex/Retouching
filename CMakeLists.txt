cmake_minimum_required(VERSION 3.21)

project(Retouching)


add_subdirectory(external/dlib)
add_subdirectory(external/tinyspline)
find_package(OpenCV 4 REQUIRED)

add_executable(Retouching src/main.cpp)
target_link_libraries(Retouching PRIVATE dlib::dlib ${OpenCV_LIBS} tinysplinecxx)
target_compile_features(Retouching PRIVATE $<IF:$<PLATFORM_ID:Windows>,cxx_std_23,cxx_std_20>) # to enable `/std:c++latest` 


