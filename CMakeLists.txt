cmake_minimum_required (VERSION 3.1)
set (CMAKE_CXX_STANDARD 11)
project (nnprobe)
enable_testing()

# add source directory
include_directories(${PROJECT_SOURCE_DIR}/src)

# add subdirectories
add_subdirectory(src)
add_subdirectory(tests)

