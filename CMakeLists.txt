cmake_minimum_required(VERSION 3.10)

project(ImageDisplay LANGUAGES CXX CUDA)

set(CMAKE_INCLUDE_CURRENT_DIR ON)

# configure file main.cpp and set variable SRC_DIR to the current directory
set(SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR})
configure_file(curdir.h.in ${SRC_DIR}/curdir.h @ONLY)

add_executable(main main.cpp)

set_source_files_properties(main.cpp PROPERTIES LANGUAGE CUDA)