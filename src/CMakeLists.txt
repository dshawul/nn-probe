cmake_minimum_required(VERSION 3.1)
set (CMAKE_CXX_STANDARD 11)

#
# Add sources and our shared library target, define macros
#

file(GLOB SOURCES "*.cpp")

add_library(nnprobe SHARED ${SOURCES})
if(TRT)
    add_executable(device device.cpp)
endif()

target_compile_definitions(nnprobe PRIVATE 
                         PARALLEL=1 
                         USE_SPINLOCK=1
                         ARC_64BIT=1
                         HAS_POPCNT=1
                         HAS_PREFETCH=1)

#
# Options for our neural network library
#

option(TENSORFLOW  "Build with tensorflow lib." ON)
option(TENSORFLOW_CC "Build with tensorflow_cc lib" OFF)
option(TF_SHARED "Build with shared tensorflow lib" ON)
option(TRT "Build with TensorRT backend" OFF)

#
# Tensorflow backend
#   option 1. Use tensorflow_cc library
#   option 2. Use manually built tensorflow library
#
if(TENSORFLOW OR TENSORFLOW_CC)

target_compile_definitions(nnprobe PRIVATE TENSORFLOW=1)

if(TENSORFLOW_CC)
    set(TF_DIR "$ENV{TENSORFLOW_CC_ROOT}")
    set(TF_DIR_INC "${TF_DIR}/include/tensorflow")
    set(TF_DIR_LIB "${TF_DIR}/lib/tensorflow_cc")
else()
    set(TF_DIR "$ENV{TENSORFLOW_ROOT}")
    set(TF_DIR_INC "${TF_DIR}")
    set(TF_DIR_LIB "${TF_DIR}/bazel-bin/tensorflow")
endif()

target_include_directories(
    nnprobe PUBLIC
    "${TF_DIR_INC}"
    "${TF_DIR_INC}/tensorflow/contrib/makefile/downloads"
    "${TF_DIR_INC}/tensorflow/contrib/makefile/downloads/eigen"
    "${TF_DIR_INC}/tensorflow/contrib/makefile/downloads/gemmlowp"
    "${TF_DIR_INC}/tensorflow/contrib/makefile/downloads/nsync/public"
    "${TF_DIR_INC}/tensorflow/contrib/makefile/gen/protobuf-host/include"
)

if(TF_SHARED)
    target_include_directories(nnprobe PUBLIC "${TF_DIR_INC}/bazel-genfiles")

    find_library(TFCC tensorflow_cc HINTS "${TF_DIR_LIB}")
    target_link_libraries(nnprobe PUBLIC ${TFCC})
else()
    target_include_directories(
        nnprobe PUBLIC
        "${TF_DIR_INC}/tensorflow/contrib/makefile/gen/host_obj"
        "${TF_DIR_INC}/tensorflow/contrib/makefile/gen/proto"
        "${TF_DIR_INC}/tensorflow/contrib/makefile/gen/proto_text"
    )

    if(TENSORFLOW_CC)
        find_library(NSYNC nsync HINTS "${TF_DIR_LIB}")
        find_library(TFCORE tensorflow-core HINTS "${TF_DIR_LIB}")
    else()
        find_library(NSYNC nsync HINTS "${TF_DIR}/tensorflow/contrib/makefile/downloads/nsync/builds/default.linux.c++11")
        find_library(TFCORE tensorflow-core HINTS "${TF_DIR}/tensorflow/contrib/makefile/gen/lib")
    endif()
    target_link_libraries(nnprobe PUBLIC "${NSYNC}" )
    target_link_libraries(nnprobe PUBLIC "-Wl,--whole-archive ${TFCORE} -Wl,--no-whole-archive" )
endif()

if(TENSORFLOW_CC)
    find_library(PROTOBUF protobuf HINTS "${TF_DIR_LIB}")
    target_link_libraries(nnprobe PUBLIC "${PROTOBUF}" )
else()
    find_library(PROTOBUF protobuf HINTS "${TF_DIR}/tensorflow/contrib/makefile/gen/protobuf-host/lib")
    target_link_libraries(nnprobe PUBLIC "${PROTOBUF}" )
endif()

endif()

#
# TensorRT backend
#

if(TRT)
    set(TRT_DIR "$ENV{TensorRT_ROOT}")

    find_package(CUDA REQUIRED)
    set(CUDA_DIR "${CUDA_TOOLKIT_ROOT_DIR}")

    target_compile_definitions(nnprobe PRIVATE TRT=1)

    target_include_directories(
        nnprobe PUBLIC
        "${TRT_DIR}/include"
        "${CUDA_DIR}/include"
    )
    target_include_directories(
        device PUBLIC
        "${TRT_DIR}/include"
        "${CUDA_DIR}/include"
    )
    find_library(CUDART cudart HINTS "${CUDA_DIR}/lib64")
    target_link_libraries(device PUBLIC "${CUDART}" )
    target_link_libraries(nnprobe PUBLIC "${CUDART}" )
    find_library(NVINFER nvinfer HINTS "${TRT_DIR}/lib")
    target_link_libraries(nnprobe PUBLIC "${NVINFER}" )
    find_library(NVONNXPARSER nvonnxparser HINTS "${TRT_DIR}/lib")
    target_link_libraries(nnprobe PUBLIC "${NVONNXPARSER}" )
endif()

#
# Installation target
#
install (TARGETS nnprobe DESTINATION lib)
install (FILES nnprobe.h DESTINATION include)

