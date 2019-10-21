#ifndef __NNPROBE__
#define __NNPROBE__

#include <stdint.h>

enum {_CPU, _GPU};
enum {_FLOAT, _HALF, _INT8};

#if defined (_WIN32)
#   define _CDECL __cdecl
#ifdef DLL_EXPORT
#   define DLLExport extern "C" __declspec(dllexport)
#else
#   define DLLExport extern "C" __declspec(dllimport)
#endif
#else
#   define _CDECL
#   define DLLExport extern "C"
#endif

/**
 Load neural network on CPU or GPU
 */
DLLExport void _CDECL load_neural_network(
    char* path,                   /** path to neural network.
                                      *.pb extenstion uses tensorflow backend.
                                      *.trt extension uses TensorRT backend.*/
    char* input_names,            /** String containing names of input nodes */
    char* output_names,           /** String containing names of output nodes */
    char* input_shapes,           /** String containing shapes of input nodes */
    char* output_sizes,           /** String containing shapes of output nodes */

    int nn_cache_size = 4194304,  /** neural network cache size in bytes */
    int dev_type = _CPU,          /** The type of device (CPU or GPU */ 
    int n_devices = 1,            /** number of devices to use */
    int max_threads = 1,          /** maximum number of threads simultaneoulsy probing neural network.
                                      This value is used to set the batch size. */
    int float_type = _FLOAT,      /** Precision type for nn evaluation: 0=full 1=half 2=int8 precisions */
    int delay = 0,                /** Threads are kept spinning in a loop with sleeping dely ms every iteration 
                                    It maybe useful to set this value to 1 or more, when the number of
                                    threads is much more than number of cpu cores. */
    int nn_id = 0                 /** identifier in case of multiple nets */
);
/**
 Probe neural network and return result in output buffer.
 This method will not return until all active threads call this function.
 The maximum number of threads is set when we load the neural network.
 To return results before maximum number of threads is reached, set the
 number of maximum active threads with @set_num_active_searchers.
 */
DLLExport void _CDECL probe_neural_network(
    float** inputs,                 /** multiple input buffers for neural network */
    float** outputs,                /** mulitple output buffers for nerual network */
    int* out_size,                  /** size of output buffers */

    unsigned short** out_index = 0, /** mulitple short arrays for fetching a subset of the outputs, 
                                        set to null (0) to get all output */ 
    uint64_t hash_key = 0,          /** 64-bit key used for caching neural network */
    bool hard_probe = false,        /** Do not return cached value if set to true */
    int nn_id = 0                   /** identifier in case of multiple nets */
);
/**
 Set the number of active threads that will request nn evaluation.
 This method is used to prevent hanging of @probe_neural_network
 in situations where not all threads are actively searching.
*/
DLLExport void _CDECL set_num_active_searchers(
    int n_searchers               /** number of threads currently probing the neural network. */
);

#undef _CDECL
#endif

