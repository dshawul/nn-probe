
# What is it

nn-probe is "sort of" of a wrapper over neural network libraries that provides mulit-threaded
batching and caching support.

# How to build and test

It depends on either TensorFlow and/or TensorRT for GPU support. So you need tensorflow built
first, and TensorRT libs downloaded/installed. These can be download from [NVIDA developer page](https://developer.nvidia.com/)
Make sure you download compatible versions (e.g. cuDNN 7.3 + CUDA 10.0 + TensorRT 5.0)

Clone

	git clone https://github.com/dshawul/nn-probe.git

Then

    mkdir -p build && cd build

For CPU you can use self-built Tensorflow lib or tensorflow_cc

    TENSORFLOW_ROOT=~/tensorflow cmake ..

And if you prefer to use tensorflow_cc

    TENSORFLOW_CC_ROOT=/usr/local/lib/tensorflow_cc cmake -DTENSORFLOW_CC=on -DTENSORFLOW=off ..

To build for GPU specify TensorRT and cuda paths

    TensorRT_ROOT=~/TensorRT-5.0.0.10 cmake -DTENSORFLOW=off -DTRT=on ..    

And if CUDA can not be found specify CUDA_TOOLKIT_ROOT_DIR as well. 

To build and install
    
    make && make install

To test

    ctest

# How to use the library

Using the library is easy. Follow the examples provided in tests/ directory.
Include the library header "nnprobe.h" and start using its load, probe and set_num_threads functions.
