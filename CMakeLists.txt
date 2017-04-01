cmake_minimum_required(VERSION 3.6)
project(AlirezaNet)

set(CMAKE_CXX_STANDARD 14)
include_directories("eigen/")
set(SOURCE_FILES test.cpp)
add_executable(AlirezaNet ${SOURCE_FILES})