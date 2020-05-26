#include <stdio.h>
#include <string.h>
#include "include/cuda_runtime_api.h"
#include "include/NvInfer.h"

void display_device_properties(int count, cudaDeviceProp* props) {
    for (int i=0; i< count; i++) {
        cudaDeviceProp& prop = props[i];
        printf("====================== GPU %d ==========================\n", i);
        printf( " --- General Information ---\n");
        printf( "Name: %s\n", prop.name );
        printf( "Compute capability: %d.%d\n", prop.major, prop.minor );
        printf( "Clock rate: %d\n", prop.clockRate );
        printf( "Device copy overlap: " );
        if (prop.deviceOverlap)
            printf( "Enabled\n" );
        else
            printf( "Disabled\n" );
        printf( "Kernel execition timeout : " );
        if (prop.kernelExecTimeoutEnabled)
            printf( "Enabled\n" );
        else
            printf( "Disabled\n" );
        printf( " --- Memory Information ---\n");
        printf( "Total global mem: %ld\n", prop.totalGlobalMem );
        printf( "Total constant Mem: %ld\n", prop.totalConstMem );
        printf( "Max mem pitch: %ld\n", prop.memPitch );
        printf( "Texture Alignment: %ld\n", prop.textureAlignment );
        printf( " --- MP Information ---\n");
        printf( "Multiprocessor count: %d\n",
                prop.multiProcessorCount );
        printf( "Shared mem per mp: %ld\n", prop.sharedMemPerBlock );
        printf( "Registers per mp: %d\n", prop.regsPerBlock );
        printf( "Threads in warp: %d\n", prop.warpSize );
        printf( "Max threads per block: %d\n",
                prop.maxThreadsPerBlock );
        printf( "Max thread dimensions: (%d, %d, %d)\n",
                prop.maxThreadsDim[0], prop.maxThreadsDim[1],
                prop.maxThreadsDim[2] );
        printf( "Max grid dimensions: (%d, %d, %d)\n",
                prop.maxGridSize[0], prop.maxGridSize[1],
                prop.maxGridSize[2] );
    }
    printf("=======================================================\n");
}

using namespace nvinfer1;
class Logger : public ILogger {
    void log(Severity severity, const char* msg) override {
    }
};

int main(int argc, char** argv) {
    int count;
    cudaGetDeviceCount( &count );
    cudaDeviceProp* props = new cudaDeviceProp[count];
    for (int i = 0; i < count; i++)
        cudaGetDeviceProperties( &props[i], i );

    Logger logger;
    
    if(argc == 1)
        display_device_properties(count, props);
    else if(!strcmp(argv[1],"-n") || !strcmp(argv[1],"--number"))
        printf("%d\n",count);
    else if(!strcmp(argv[1],"--mp")) {
        int sum = 0;
        for (int i = 0; i < count; i++)
            sum += props[i].multiProcessorCount;
        printf("%d\n",sum);
    }
    else if(!strcmp(argv[1],"--mp-each")) {
        for (int i = 0; i < count; i++)
            printf("%d ",props[i].multiProcessorCount);
        printf("\n");
    }
    else if(!strcmp(argv[1],"--name"))
        printf("%s\n",props[0].name);
    else if(!strcmp(argv[1],"--name-each")) {
        for (int i = 0; i < count; i++)
            printf("%s\n",props[i].name);
    }
    else if(!strcmp(argv[1],"--int8")) {
        for (int i = 0; i < count; i++) {
            cudaSetDevice(i);
            IBuilder* builder = createInferBuilder(logger);
            if(!builder->platformHasFastInt8()) {
                printf("N\n");
                builder->destroy();
                return 0;
            }
            builder->destroy();
        }
        printf("Y\n");
    } else if(!strcmp(argv[1],"--int8-each")) {
        for (int i = 0; i < count; i++) {
            cudaSetDevice(i);
            IBuilder* builder = createInferBuilder(logger);
            if(builder->platformHasFastInt8())
                printf("Y");
            else
                printf("N");
            builder->destroy();
        }
        printf("\n");
    } else if(!strcmp(argv[1],"--fp16")) {
        for (int i = 0; i < count; i++) {
            cudaSetDevice(i);
            IBuilder* builder = createInferBuilder(logger);
            if(!builder->platformHasFastFp16()) {
                printf("N");
                builder->destroy();
                return 0;
            }
            builder->destroy();
        }
        printf("Y\n");
    } else if(!strcmp(argv[1],"--fp16-each")) {
        for (int i = 0; i < count; i++) {
            cudaSetDevice(i);
            IBuilder* builder = createInferBuilder(logger);
            if(builder->platformHasFastFp16())
                printf("Y");
            else
                printf("N");
            builder->destroy();
        }
        printf("\n");
    } else if(!strcmp(argv[1],"--help") || !strcmp(argv[1],"-h")) {
        printf("Usage: device [option]\n"
               "   -h,--help    Display this help message\n"
               "   -n,--number  Number of GPUs\n"
               "   --mp         Total number of multiprocessors\n"
               "   --mp-each    Number of multiprocessors for each GPU\n"
               "   --name       Name of GPU 0\n"
               "   --name-each  Name of each GPU\n"
               "   --int8       Does all GPUs support fast INT8?\n"
               "   --int8-each  INT8 support for each GPU\n"
               "   --fp16       Does all GPUs support fast FP16?\n"
               "   --fp16-each  FP16 support for each GPU\n"
              );
    }
  
    return 0;
}
