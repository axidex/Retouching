cmake_minimum_required(VERSION 3.21)

project(Retouching)


add_subdirectory(external/dlib)
add_subdirectory(external/tinyspline)
add_subdirectory(external/wavelib)
find_package(OpenCV 4 REQUIRED)
set(FILTER_PUBLIC_SRC
    src/newfilter.cpp
)

add_library(Filter STATIC ${FILTER_PUBLIC_SRC})
target_link_libraries(Filter PRIVATE ${OpenCV_LIBS})
set_target_properties(Filter PROPERTIES LINKER_LANGUAGE CXX)

target_include_directories(Filter PUBLIC src/)


add_executable(Retouching src/main.cpp)
target_link_libraries(Retouching PRIVATE dlib::dlib ${OpenCV_LIBS} tinysplinecxx wavelib Filter)
target_include_directories(Retouching PRIVATE external/wavelib/header)
target_compile_features(Retouching PRIVATE $<IF:$<PLATFORM_ID:Windows>,cxx_std_23,cxx_std_20>) # to enable `/std:c++latest` 