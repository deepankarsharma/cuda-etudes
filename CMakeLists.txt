cmake_minimum_required(VERSION 3.12)
project(imgui_bundle_example_integration LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 20)


if (NOT TARGET imgui_bundle)
    message(STATUS "Fetching imgui_bundle...")
    include(FetchContent)
    Set(FETCHCONTENT_QUIET FALSE)
    FetchContent_Declare(imgui_bundle GIT_REPOSITORY https://github.com/pthom/imgui_bundle.git GIT_TAG "v1.3.0")
    FetchContent_MakeAvailable(imgui_bundle)
endif()


imgui_bundle_add_app(imgui_bundle_example_integration hello_world.cpp)
set_source_files_properties(hello_world.cpp PROPERTIES LANGUAGE CUDA)

