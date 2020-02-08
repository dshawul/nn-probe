#!/bin/bash

set -eux

OS=ubuntu
export TENSORFLOW_ROOT=${TENSORFLOW_ROOT:-~/tensorflow}
export TensorRT_ROOT=${TensorRT_ROOT:-~/TensorRT-6.0.1.5-cuda100}
export CUDA_TOOLKIT_ROOT_DIR=${CUDA_TOOLKIT_ROOT_DIR:-/usr/local/cuda}

mkdir -p nnprobe-${OS}-cpu
mkdir -p nnprobe-${OS}-gpu

##################
# build for GPU
##################
[ -d "build" ] && rm -rf build
mkdir -p build && cd build
cmake -DTENSORFLOW=off -DTRT=on ../src
make
cp libnnprobe.so ../nnprobe-${OS}-gpu/libnnprobe.so
cd ..

DLL=`ldd nnprobe-${OS}-gpu/libnnprobe.so | awk '{ print $3 }' | grep libnv`
if ! [ -z "$DLL" ]; then
    cp $DLL nnprobe-${OS}-gpu
fi

set +e
DLL=`ldd nnprobe-${OS}-gpu/libnnprobe.so | awk '{ print $3 }' | grep libcu`
set -e
if ! [ -z "$DLL" ]; then
    cp $DLL nnprobe-${OS}-gpu
fi

#######################
# build for CPU
#######################
mkdir -p $TENSORFLOW_ROOT/tensorflow/cc/nnprobe/
cp src/*.cpp src/*.h src/BUILD $TENSORFLOW_ROOT/tensorflow/cc/nnprobe/
cd $TENSORFLOW_ROOT
bazel build --config=opt --config=monolithic //tensorflow/cc/nnprobe:libnnprobe.so
cd -
cp $TENSORFLOW_ROOT/bazel-bin/tensorflow/cc/nnprobe/libnnprobe.so nnprobe-${OS}-cpu/libnnprobe.so
chmod 755 nnprobe-${OS}-cpu/libnnprobe.so

zip -r nnprobe-${OS}-cpu.zip nnprobe-${OS}-cpu
zip -r nnprobe-${OS}-gpu.zip nnprobe-${OS}-gpu
