cmake_minimum_required(VERSION 2.8)
project( DisplayImage )
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )
add_executable( svmpredicker svmpredicker.cpp )
target_link_libraries( svmpredicker ${OpenCV_LIBS} )
add_executable( svmtrainer svmtrainer.cpp )
target_link_libraries( svmtrainer ${OpenCV_LIBS} )



