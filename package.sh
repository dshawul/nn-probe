#!/bin/bash

set -eux

OS=ubuntu
export TENSORFLOW_ROOT=${TENSORFLOW_ROOT:-~/tensorflow}
export TensorRT_ROOT=${TensorRT_ROOT:-~/TensorRT-7.2.0.14-cuda110}
export CUDA_ROOT=${CUDA_ROOT:-/usr/local/cuda-11.0}
export CUDNN_ROOT=${CUDNN_ROOT:-~/cudnn-804}
export LD_LIBRARY_PATH=$CUDNN_ROOT/lib64:$CUDA_ROOT/lib64::$TensorRT_ROOT/lib

##################
# build for GPU
##################
mkdir -p nnprobe-${OS}-gpu
[ -d "build" ] && rm -rf build
mkdir -p build && cd build
cmake -DTENSORFLOW=off -DTRT=on ../src
make
cp libnnprobe.so ../nnprobe-${OS}-gpu/libnnprobe.so
cp device ../nnprobe-${OS}-gpu/device
cd ..

DLL=`ldd nnprobe-${OS}-gpu/libnnprobe.so | awk '{ print $3 }' | grep TensorRT`
if ! [ -z "$DLL" ]; then
    cp $DLL nnprobe-${OS}-gpu
fi

set +e
DLL=`ldd nnprobe-${OS}-gpu/libnnprobe.so | awk '{ print $3 }' | grep libcu`
set -e
if ! [ -z "$DLL" ]; then
    cp $DLL nnprobe-${OS}-gpu
fi

zip -r nnprobe-${OS}-gpu.zip nnprobe-${OS}-gpu

#######################
# build for CPU
#######################
mkdir -p nnprobe-${OS}-cpu
mkdir -p $TENSORFLOW_ROOT/tensorflow/cc/nnprobe/
cp src/*.cpp src/*.h src/BUILD $TENSORFLOW_ROOT/tensorflow/cc/nnprobe/
cd $TENSORFLOW_ROOT
bazel build --config=opt --config=monolithic //tensorflow/cc/nnprobe:libnnprobe.so
cd -
cp $TENSORFLOW_ROOT/bazel-bin/tensorflow/cc/nnprobe/libnnprobe.so nnprobe-${OS}-cpu/libnnprobe.so
chmod 755 nnprobe-${OS}-cpu/libnnprobe.so

zip -r nnprobe-${OS}-cpu.zip nnprobe-${OS}-cpu
