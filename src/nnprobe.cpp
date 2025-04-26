#include <vector>
#include <math.h>
#include <iostream>

#ifdef TENSORFLOW
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/graph/default_device.h"
#endif

#ifdef TRT
#include <fstream>
#include <string>
#include <iterator>
#include <tuple>
#include <cstring>
#include "cuda_runtime_api.h"
#include "NvInfer.h"
#include "NvOnnxParser.h"
#endif

#define DLL_EXPORT
#include "nnprobe.h"
#undef DLL_EXPORT

#include "my_types.h"

//#define USE_ZERO_COPY

static int N_DEVICES;
static int n_searchers;
static std::atomic_int n_active_searchers;
static std::atomic_int chosen_device = {0};
static int delayms = 0;
static int batch_size_factor = 0;

enum { FCFS, ROUNDROBIN };
static int scheduling = FCFS;

enum { FP32 = 0, FP16, FP8 };
static const char* float_type_string[] = {"FLOAT", "HALF", "INT8"};

/*
Check fp16 and int8 support
*/
static bool hasFastFp16(int device = 0)
{
    int major = 0, minor = 0;
    cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device);
    cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device);
    return (major >= 7);
}

static bool hasFastInt8(int device = 0)
{
    int major = 0, minor = 0;
    cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device);
    cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device);
    return (major > 7 || (major == 7 && minor >= 5));
}

/*
Neural network properties
*/
struct NeuralNet {
    std::vector<std::string> input_layer_names;
    std::vector<std::string> output_layer_names;
    std::vector<std::tuple<int,int,int>> input_layer_shapes;
    std::vector<int> output_layer_sizes;
    char path[256];

    uint64_t* nn_cache;
    uint32_t nn_cache_mask;
    uint32_t hash_entry_sz;

    void allocate_nn_cache(uint32_t sizeb);
    void store_nn_cache(const uint64_t hash_key,  unsigned short** const p_index,
                           int* const p_size, float** const p_outputs);
    bool retrieve_nn_cache(const uint64_t hash_key, unsigned short** const p_index,
                              int* const p_size, float** p_outputs);
};

/*
  Network model
*/
class Model {
public:
    NeuralNet* pnn;
    float*** p_outputs;
    unsigned short*** p_index;
    int** p_size;
    std::atomic_int n_batch;
    std::atomic_int n_batch_i;
    std::atomic_int n_batch_eval;
    std::atomic_int n_finished_threads;
    int float_type;
    int BATCH_SIZE;
    int id;
    Model(NeuralNet* pnn_, int batch_size_, int float_type_) {
        pnn = pnn_;
        BATCH_SIZE = batch_size_;
        float_type = float_type_;
        const int NOUT = pnn->output_layer_names.size();
        p_outputs = new float**[NOUT];
        p_index = new unsigned short**[NOUT];
        p_size = new int*[NOUT];
        for(int i = 0; i < NOUT; i++) {
            p_outputs[i] = new float*[BATCH_SIZE];
            p_index[i] = new unsigned short*[BATCH_SIZE];
            p_size[i] = new int[BATCH_SIZE];
        }
        n_batch = 0;
        n_batch_i = 0;
        n_batch_eval = 0;
        n_finished_threads = 0;
        id = 0;
    }
    ~Model() {
    }
    virtual float* get_input_buffer(int) = 0;
    virtual int get_input_size(int) = 0;
    virtual void predict() = 0;
    virtual void LoadGraph(int, int) = 0;
    static int dev_type;
    /*conditional wait on evaluation*/
    MUTEX wait_eval_lock;
    COND  wait_eval_cond;

    void signal_eval() {
        std::lock_guard<std::mutex> lock(wait_eval_lock);
        c_signal(wait_eval_cond);
    }
    void wait_eval() {
        std::unique_lock<std::mutex> lk(wait_eval_lock);
        c_wait(wait_eval_cond, lk,
            [this]{return (n_finished_threads != 0);});
    }
    /*conditional wait on getting resources*/
    static std::atomic_int n_idle_gpus;
    static MUTEX wait_gpu_lock;
    static COND  wait_gpu_cond;

    static void signal_gpu() {
        std::lock_guard<std::mutex> lock(wait_gpu_lock);
        c_signal(wait_gpu_cond);
    }
    static void wait_gpu() {
        std::unique_lock<std::mutex> lk(wait_gpu_lock);
        c_wait(wait_gpu_cond, lk,
            []{return n_idle_gpus != 0;});
    }
};

int Model::dev_type;

std::atomic_int Model::n_idle_gpus;
MUTEX Model::wait_gpu_lock;
COND Model::wait_gpu_cond;

static Model** netModel[16];

/*
  TensorFlow model
*/
#ifdef TENSORFLOW

using namespace tensorflow;

class TfModel : public Model {
    Tensor** input_layers;
    Session* session;
    std::vector<std::pair<std::string, Tensor> > inps;
    std::vector<std::string> outs;
public:
    TfModel(NeuralNet*, int, int);
    ~TfModel();
    void LoadGraph(int dev_id, int dev_type);
    void predict();
    float* get_input_buffer(int idx) {
        return (float*)(input_layers[idx]->tensor_data().data());
    }
    int get_input_size(int idx) {
        return input_layers[idx]->NumElements() / BATCH_SIZE;
    }
};

TfModel::TfModel(NeuralNet* pnn_, int bsize, int float_type) : Model(pnn_, bsize, float_type) {
    input_layers = new Tensor*[pnn->input_layer_names.size()];
}
TfModel::~TfModel() {
    for(int n = 0; n < pnn->input_layer_names.size(); n++) {
        delete[] input_layers[n];
    }
    delete[] input_layers;
}
void TfModel::LoadGraph(int dev_id, int dev_type) {
    Model::id = dev_id;

    GraphDef graph_def;
    Status load_graph_status =
        ReadBinaryProto(Env::Default(), pnn->path, &graph_def);

    std::string dev_name = ((dev_type == _GPU) ? "/gpu:" : "/cpu:") + std::to_string(dev_id);
    graph::SetDefaultDevice(dev_name, &graph_def);
    printf("Loading graph on %s\n",dev_name.c_str());
    fflush(stdout);

    for (auto &node: *graph_def.mutable_node()) {
        for(int n = 0; n < pnn->input_layer_names.size(); n++) {
            if(node.name() == pnn->input_layer_names[n]) {
                TensorShape nshape({BATCH_SIZE});
                auto shape = node.attr().at("shape").shape();
                std::tuple<int,int,int> shp = pnn->input_layer_shapes[n];

                if(shape.dim_size() > 2) {
                    nshape.AddDim(std::get<2>(shp));
                    nshape.AddDim(std::get<1>(shp));
                    nshape.AddDim(std::get<0>(shp));
                } else {
                    nshape.AddDim(std::get<0>(shp));
                }

                input_layers[n] = new Tensor(DT_FLOAT, nshape);

                printf("%d. %s = ", n, node.name().c_str());
                for (int i = 1; i < shape.dim_size(); i++)
                    printf("%d ",(int)shape.dim(i).size());
                printf("\n");
            }
        }
        for(int n = 0; n < pnn->output_layer_names.size(); n++) {
            if(node.name() == pnn->output_layer_names[n]) {
                printf("%d. %s", n, node.name().c_str());
                printf("\n");
            }
        }
    }
    fflush(stdout);
    

#if 0
    std::cout << "=============================" << std::endl;
    for (auto &node: *graph_def.mutable_node())
        std::cout << node.name() << std::endl;
    std::cout << "=============================" << std::endl;
#endif

    SessionOptions options;
    Status status = NewSession(options, &session);
    session->Create(graph_def);

    for(int n = 0; n < pnn->input_layer_names.size(); n++) {
        std::pair<std::string, Tensor> pr( 
            pnn->input_layer_names[n], *(input_layers[n]) );
        inps.push_back(pr);
    }
    for(int n = 0; n < pnn->output_layer_names.size(); n++)
        outs.push_back(pnn->output_layer_names[n]);
}

void TfModel::predict() {
    std::vector<Tensor> outputs;
    TF_CHECK_OK( session->Run(inps, outs, {}, &outputs) );

    for(int k = 0; k < pnn->output_layer_names.size(); k++) {
        auto pp = outputs[k].matrix<float>();

        if(p_index[k][0] == 0) {
            for(int i = 0;i < n_batch; i++) {
                for(int j = 0;j < p_size[k][i];j++) {
                    p_outputs[k][i][j] = pp(i,j);
                }
            }
        } else {
            for(int i = 0;i < n_batch; i++) {
                for(int j = 0;j < p_size[k][i];j++) {
                    int idx = p_index[k][i][j];
                    p_outputs[k][i][j] = pp(i,idx);
                }
            }
        }
    }
}
#endif

/*
  TensorRT model
*/
#ifdef TRT

using namespace nvinfer1;

class Int8CacheCalibrator : public IInt8EntropyCalibrator2 {
  
public:

    Int8CacheCalibrator(NeuralNet* pnn_, int float_type_) {
        pnn = pnn_;
        float_type = float_type_;

        if(float_type == FP8) {

            void* buf;
            for(int n = 0; n < pnn->input_layer_names.size(); n++) {
                size_t sz = std::get<0>(pnn->input_layer_shapes[n]) *
                            std::get<1>(pnn->input_layer_shapes[n]) *
                            std::get<2>(pnn->input_layer_shapes[n]);
                cudaMalloc(&buf, CAL_BATCH_SIZE * sizeof(float) * sz);
                buffers.push_back(buf);
                buf = (float*) malloc(CAL_BATCH_SIZE * sizeof(float) * sz);
                buffers_h.push_back(buf);
            }

            counter = 0;

            epd_file = fopen(calib_file_name.c_str(),"rb");
            if(!epd_file) {
                printf("Calibration file not found!\n");
                fflush(stdout);
                exit(0);
            }
        }
    }

    ~Int8CacheCalibrator() override {
        if(float_type == FP8) {
            for(int n = 0; n < pnn->input_layer_names.size(); n++) {
                cudaFree(buffers[n]);
                free(buffers_h[n]);
            }
            if(epd_file)
                fclose(epd_file);
        }
    }

    int getBatchSize() const noexcept override {
        return CAL_BATCH_SIZE;
    }

    bool getBatch(void* bindings[], const char* names[], int nbBindings) noexcept override {
        if (counter >= NUM_CAL_BATCH)
            return false;

        std::cout << "Calibrating on Batch " << counter + 1 << " of " << NUM_CAL_BATCH << "\r";

        for(int i = 0; i < CAL_BATCH_SIZE; i++) {
            for(int n = 0; n < pnn->input_layer_names.size(); n++) {
                size_t sz = std::get<0>(pnn->input_layer_shapes[n]) *
                            std::get<1>(pnn->input_layer_shapes[n]) *
                            std::get<2>(pnn->input_layer_shapes[n]);
                float* p = ((float*)buffers_h[n]) + i * sz;
                fread(p, 1, sizeof(float) * sz, epd_file);
            }
        }

        for(int n = 0; n < pnn->input_layer_names.size(); n++) {
            size_t sz = std::get<0>(pnn->input_layer_shapes[n]) *
                        std::get<1>(pnn->input_layer_shapes[n]) *
                        std::get<2>(pnn->input_layer_shapes[n]);
            cudaMemcpy(buffers[n], buffers_h[n], 
                CAL_BATCH_SIZE * sizeof(float) * sz, cudaMemcpyHostToDevice);
            bindings[n] = buffers[n];
        }

        counter++;
        return true;
    }

    const void* readCalibrationCache(size_t& length) noexcept override {
        return nullptr;
    }

    void writeCalibrationCache(const void* cache, size_t length) noexcept override {
    }

private:
    NeuralNet* pnn;
    int float_type;
    std::vector<void*> buffers;
    std::vector<void*> buffers_h;
    int counter;
    FILE* epd_file;
    static const int CAL_BATCH_SIZE = 256;
    static const int NUM_CAL_BATCH = 10;
    static const std::string calib_file_name;
};

const std::string Int8CacheCalibrator::calib_file_name = "calibrate.dat";

class Logger : public ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        if (severity != Severity::kINFO &&
            severity != Severity::kVERBOSE &&
            severity != Severity::kWARNING)
                std::cout << msg << std::endl;
    }
};

class TrtModel : public Model {
    ICudaEngine* engine;
    IExecutionContext* context;
    Logger logger;
    int numBindings;
    std::vector<void*> buffers;
    std::vector<float*> buffers_h;
    std::vector<int> buffer_sizes;
    std::vector<int> inp_index;
    std::vector<int> out_index;
    size_t batch_size_bytes;
public:
    TrtModel(NeuralNet*, int, int);
    ~TrtModel();
    void LoadGraph(int dev_id, int dev_type);
    void predict();
    float* get_input_buffer(int idx) {
        return buffers_h[inp_index[idx]];
    }
    int get_input_size(int idx) {
        return buffer_sizes[inp_index[idx]];
    }
};

TrtModel::TrtModel(NeuralNet* pnn_, int bsize, int float_type) : Model(pnn_, bsize, float_type) {
    context = 0;
    engine = 0;
    numBindings = 0;
}

TrtModel::~TrtModel() {
#ifndef USE_ZERO_COPY
    cudaFree(buffers[0]);
#endif
    cudaFreeHost(buffers_h[0]);
}

void TrtModel::LoadGraph(int dev_id, int dev_type) {
    std::string dev_name = ((dev_type == _GPU) ? "/gpu:" : "/cpu:") + std::to_string(dev_id);
    printf("Loading graph on %s\n",dev_name.c_str());
    fflush(stdout);

    Model::id = dev_id;
    cudaSetDevice(Model::id);

    std::string trtName = std::string(pnn->path) + "." + 
                          std::to_string(BATCH_SIZE)+ "_" + 
                          std::to_string(float_type) +
                          ".trt";
    std::ifstream ifs(trtName.c_str(), std::ios::in | std::ios::binary);

    /*generate or read trt file*/
    if (!ifs.is_open()) {

        /*if requested precision is not supported, fallback to next available*/
        IBuilder* builder = createInferBuilder(logger);
        if(float_type == FP8 && !hasFastInt8(dev_id)) {
            if(hasFastFp16(dev_id)) {
                float_type = FP16;
            } else {
                float_type = FP32;
            }
            printf("Switching to \"%s\" precision for GPU %d\n",
                float_type_string[float_type], dev_id);
            fflush(stdout);
        }
        if(float_type == FP16 && !hasFastFp16(dev_id)) {
            if(hasFastInt8(dev_id)) {
                float_type = FP8;
            } else {
                float_type = FP32;
            }
            printf("Switching to \"%s\" precision for GPU %d\n",
                float_type_string[float_type], dev_id);
            fflush(stdout);
        }
        
        /*create network*/
        const auto explicitBatch =
            (1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH));
        INetworkDefinition* network = builder->createNetworkV2(explicitBatch);

        /*read network uff/onnx*/
        IBuilderConfig* config = builder->createBuilderConfig();
        IOptimizationProfile* profile = builder->createOptimizationProfile();
        {
            nvonnxparser::IParser* parser = nvonnxparser::createParser(*network, logger);

            for(int n = 0; n < pnn->input_layer_names.size(); n++) {
                nvinfer1::Dims4 mydim(
                    BATCH_SIZE,
                    std::get<0>(pnn->input_layer_shapes[n]),
                    std::get<1>(pnn->input_layer_shapes[n]),
                    std::get<2>(pnn->input_layer_shapes[n]));
                auto name = pnn->input_layer_names[n].c_str();

                profile->setDimensions(name, OptProfileSelector::kMIN, mydim);
                profile->setDimensions(name, OptProfileSelector::kOPT, mydim);
                profile->setDimensions(name, OptProfileSelector::kMAX, mydim);
            }

            if(!parser->parseFromFile(pnn->path,
                static_cast<int32_t>(ILogger::Severity::kWARNING))) {
                std::cout << "Fail to parse network " << pnn->path << std::endl;
                return;
            }

            config->addOptimizationProfile(profile);
        }

        /*create engine*/
        Int8CacheCalibrator calibrator(pnn, float_type);
        config->setMemoryPoolLimit(MemoryPoolType::kWORKSPACE, 1 << 30);

        if (float_type == FP16) {
            config->setFlag(BuilderFlag::kFP16);
        } else if (float_type == FP8) {
            config->setFlag(BuilderFlag::kINT8);
            config->setInt8Calibrator(&calibrator);
        }

        IHostMemory* trtModelStream = builder->buildSerializedNetwork(*network,*config);
        if (!trtModelStream) {
            std::cout << "Unable to create network" << std::endl;
            return;
        }
        IRuntime* infer = createInferRuntime(logger);
        engine = infer->deserializeCudaEngine(trtModelStream->data(), trtModelStream->size());

        FILE* out_file = fopen(trtName.c_str(),"wb");
        fwrite((char*)(trtModelStream->data()), 1, trtModelStream->size(), out_file);
        fclose(out_file);
    } else {
        char* trtModelStream{nullptr};
        size_t size{0};

        ifs.seekg(0, ifs.end);
        size = ifs.tellg();
        ifs.seekg(0, ifs.beg);
        trtModelStream = new char[size];
        ifs.read(trtModelStream, size);
        ifs.close();

        IRuntime* infer = createInferRuntime(logger);
        engine = infer->deserializeCudaEngine(trtModelStream, size);
        if (trtModelStream) delete[] trtModelStream;
    }

    context = engine->createExecutionContext();
    numBindings = engine->getNbIOTensors();
    
    std::vector<const char*> tensorNames;
    for (int i = 0; i < numBindings; i++) {
        tensorNames.push_back(engine->getIOTensorName(i));
    }

    /*Pinned memory*/
    int start = 1;
    batch_size_bytes = 0;
    for(int i = 0; i < numBindings; i++) {
        const char* tensorName = tensorNames[i];

        Dims d = engine->getTensorShape(tensorName);
        size_t size = 1;
        for(size_t j = start; j < d.nbDims; j++)
            size*= d.d[j];
        batch_size_bytes += size;
        buffer_sizes.push_back(size);

#if 1
        if(dev_id == N_DEVICES -1) {
            printf("%d. %s %d =",i,tensorName,(int)size);
            for(size_t j = start; j < d.nbDims; j++)
                printf(" %ld",d.d[j]);
            printf("\n");
            fflush(stdout);
        }
#endif
    }
    batch_size_bytes *= (BATCH_SIZE * sizeof(float));

    float* pDevice, *pHost;
#ifdef USE_ZERO_COPY
    cudaHostAlloc((void**)&pHost, 
        batch_size_bytes, 
        cudaHostAllocMapped);
    cudaHostGetDevicePointer((void**)&pDevice,(void*)pHost,0);
#else
    cudaHostAlloc((void**)&pHost, 
        batch_size_bytes, 
        cudaHostAllocDefault);
    cudaMalloc((void**)&pDevice,
        batch_size_bytes);
#endif

    for(int i = 0; i < numBindings; i++) {
        size_t size = buffer_sizes[i];
        buffers.push_back(pDevice);
        buffers_h.push_back(pHost);
        pDevice += BATCH_SIZE * size;
        pHost += BATCH_SIZE * size;
    }

    for(int n = 0; n < pnn->input_layer_names.size(); n++)
        for (int i = 0; i < tensorNames.size(); i++) {
            if (strcmp(pnn->input_layer_names[n].c_str(), tensorNames[i]) == 0) {
                inp_index.push_back(i);
                break;
            }
        }
    for(int n = 0; n < pnn->output_layer_names.size(); n++)
        for (int i = 0; i < tensorNames.size(); i++) {
            if (strcmp(pnn->output_layer_names[n].c_str(), tensorNames[i]) == 0) {
                out_index.push_back(i);
                break;
            }
        }
}
void TrtModel::predict() {

    cudaSetDevice(Model::id);

#ifndef USE_ZERO_COPY
    cudaMemcpy(buffers[0], buffers_h[0], batch_size_bytes, cudaMemcpyHostToDevice);
#endif
    context->executeV2(buffers.data());
#ifndef USE_ZERO_COPY
    cudaMemcpy(buffers_h[0], buffers[0], batch_size_bytes, cudaMemcpyDeviceToHost);
#endif

    for(int k = 0; k < pnn->output_layer_names.size(); k++) {
        int NN_MAX = buffer_sizes[out_index[k]];
        float* output = buffers_h[out_index[k]];
        
        if(p_index[k][0] == 0) {
            for(int i = 0;i < n_batch; i++) {
                for(int j = 0;j < p_size[k][i];j++) {
                    p_outputs[k][i][j] = output[i * NN_MAX + j];
                }
            }
        } else {
            for(int i = 0;i < n_batch; i++) {
                for(int j = 0;j < p_size[k][i];j++) {
                    int idx = p_index[k][i][j];
                    p_outputs[k][i][j] = output[i * NN_MAX + idx];
                }
            }
        }
    }

}
#endif

/* 
  Neural network caching
*/

void NeuralNet::allocate_nn_cache(uint32_t sizekb) {
    hash_entry_sz = sizeof(uint64_t);
    for(int k = 0; k < output_layer_sizes.size(); k++) {
        hash_entry_sz += output_layer_sizes[k] * 
                (sizeof(unsigned short) + sizeof(float));
    }
    hash_entry_sz = sizeof(uint64_t) * ( (hash_entry_sz + sizeof(uint64_t) - 1) / sizeof(uint64_t));

    uint64_t sizeb = uint64_t(sizekb) * 1024;
    uint32_t size = 1, size_max = sizeb / hash_entry_sz;
    while(2 * size <= size_max) size *= 2;
    nn_cache_mask = size - 1;
    hash_entry_sz /= sizeof(uint64_t);
    aligned_reserve<uint64_t>( nn_cache, size * hash_entry_sz );

    printf("nn_cache %d X %d = %.1f MB\n",size,int(hash_entry_sz * sizeof(uint64_t)),
        (size * hash_entry_sz * sizeof(uint64_t)) / double(1024 * 1024));
    fflush(stdout);
}

void NeuralNet::store_nn_cache(const uint64_t hash_key,  unsigned short** const p_index,
                           int* const p_size, float** const p_outputs
    ) {
    uint32_t key = uint32_t(hash_key & nn_cache_mask);
    uint64_t* const nn_hash = nn_cache + key * hash_entry_sz; 
    
    if(*nn_hash != hash_key) {
        *nn_hash = hash_key;
        uint16_t* p = (uint16_t*) (nn_hash + 1);
        for(int k = 0; k < output_layer_names.size(); k++) {
            memcpy(p, p_outputs[k], p_size[k] * sizeof(float));
            p += p_size[k] * 2;
            if(p_index[k]) {
                memcpy(p, p_index[k], p_size[k] * sizeof(uint16_t));
                p += p_size[k];
            }
        }
    }
}

bool NeuralNet::retrieve_nn_cache(const uint64_t hash_key, unsigned short** const p_index,
                              int* const p_size, float** p_outputs
    ) {
    uint32_t key = uint32_t(hash_key & nn_cache_mask);
    uint64_t* const nn_hash = nn_cache + key * hash_entry_sz;

    if(*nn_hash == hash_key) {
        uint16_t* p = (uint16_t*) (nn_hash + 1);
        for(int k = 0; k < output_layer_names.size(); k++) {
            if(p_index[k]) {
                float* const nn_outputs = (float*)p;
                p += p_size[k] * 2;
                uint16_t* const nn_index = (uint16_t*)(p);
                p += p_size[k];

                for(int i = 0; i < p_size[k]; i++) {
                    if(p_index[k][i] == nn_index[i]) {
                        p_outputs[k][i] = nn_outputs[i];
                    } else {
                        for(int j = 0; j < p_size[k]; j++) {
                            if(p_index[k][i] == nn_index[j]) {
                                p_outputs[k][i] = nn_outputs[j];
                                break;
                            }
                        }
                    }
                }
            } else {
                memcpy(p_outputs[k], p, p_size[k] * sizeof(float));
                p += p_size[k] * 2;
            }
        }

        return true;
    }
    return false;
}

/*
  Determine minibatch sizes
*/
void determine_minibatch_sizes(int t_batch_size, int* bsize) {
#ifdef TRT
    int total_mps = 0;
    for (int i = 0; i < N_DEVICES; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        bsize[i] = prop.multiProcessorCount;
        total_mps += bsize[i];
    }
    if(batch_size_factor) {
        for (int i = 0; i < N_DEVICES; i++)
            bsize[i] *= batch_size_factor;
        int tbsize = total_mps * batch_size_factor;
        if(t_batch_size % tbsize != 0) {
            printf("Number of worker threads not ideal: specified %d ideal %d\n"
               "Please use the latter for best performance\n",
               t_batch_size, (t_batch_size / tbsize) * tbsize);
            fflush(stdout);
        }
    } else {
        int tbsize = 0;
        for (int i = 0; i < N_DEVICES; i++) {
            bsize[i] = t_batch_size * bsize[i] / total_mps;
            tbsize += bsize[i];
        }
        if(tbsize != t_batch_size) {
            bsize[N_DEVICES - 1] += t_batch_size - tbsize;
            printf("MiniBatch size not ideal: specified %d ideal %d\n"
               "Please use the latter for best performance\n",
               t_batch_size, tbsize);
            fflush(stdout);
        }
    }
#else
    for (int i = 0; i < N_DEVICES; i++) {
        bsize[i] = t_batch_size / N_DEVICES;
    }
#endif
}

/*
   Initialize tensorflow
*/
static int add_to_batch(Model* net, float** iplanes, float** p_outputs,
                        int* p_size, unsigned short** p_index);

int tokenize(char *str, char** tokens, const char *str2 = " ") {
    int nu_tokens = 0;
    tokens[nu_tokens] = strtok(str, str2);
    while (tokens[nu_tokens++] != NULL) {
        tokens[nu_tokens] = strtok(NULL, str2);
    }
    return nu_tokens;
}

DLLExport void CDECL load_neural_network(
    char* path,
    char* input_names, char* output_names,
    char* input_shapes, char* output_sizes,
    int nn_cache_size, int dev_type, int n_devices,
    int max_threads, int float_type, int delay, int nn_id,
    int batch_size_factor_, int scheduling_
    ) {

#ifdef _WIN32
#   define setenv(n,v,o) _putenv_s(n,v)
#endif

    /*setenv variables*/
#ifdef TENSORFLOW
    setenv("TF_CPP_MIN_LOG_LEVEL","3",1);
#endif

    delayms = delay;
    n_searchers = max_threads;
    N_DEVICES = n_devices;
    n_active_searchers = n_searchers;
    batch_size_factor = batch_size_factor_;
    scheduling = scheduling_;
    Model::n_idle_gpus = N_DEVICES;

    NeuralNet* pnn = new NeuralNet();

    /*parse input and output node names and shapes*/
    int num_tokens;
    char buffer[4096];
    char* commands[256];

    strcpy(buffer, input_names);
    num_tokens = tokenize(buffer,commands) - 1;
    for(int i = 0; i < num_tokens; i++)
        pnn->input_layer_names.push_back(commands[i]);

    strcpy(buffer, input_shapes);
    tokenize(buffer,commands);
    for(int i = 0; i < num_tokens; i++) {
        std::tuple<int,int,int> tp(
            atoi(commands[3*i+0]),
            atoi(commands[3*i+1]),
            atoi(commands[3*i+2]) );
        pnn->input_layer_shapes.push_back(tp);
    }

    strcpy(buffer, output_names);
    num_tokens = tokenize(buffer,commands) - 1;
    for(int i = 0; i < num_tokens; i++)
        pnn->output_layer_names.push_back(commands[i]);

    strcpy(buffer, output_sizes);
    num_tokens = tokenize(buffer,commands) - 1;
    for(int i = 0; i < num_tokens; i++)
        pnn->output_layer_sizes.push_back(atoi(commands[i]));

    /*Allocate cache*/
    pnn->allocate_nn_cache(nn_cache_size);

    /*Message*/
    printf("Loading neural network : %s\n",path);
    printf("With \"%s\" precision\n", float_type_string[float_type]);
    fflush(stdout);

    /*Load tensorflow or tensorrt graphs on GPU*/
    int* minibatch = new int[N_DEVICES];
    determine_minibatch_sizes(n_searchers, minibatch);

    netModel[nn_id] = new Model*[N_DEVICES];
#if defined(TENSORFLOW) && defined(TRT)
    if(strstr(path, ".pb") != NULL) {
        for(int i = 0; i < N_DEVICES; i++)
            netModel[nn_id][i] = new TfModel(pnn, minibatch[i], float_type);
    } else if(strstr(path, ".uff") != NULL || strstr(path, ".onnx") != NULL) {
        for(int i = 0; i < N_DEVICES; i++)
            netModel[nn_id][i] = new TrtModel(pnn, minibatch[i], float_type);
    }
#elif defined(TENSORFLOW)
    for(int i = 0; i < N_DEVICES; i++)
        netModel[nn_id][i] = new TfModel(pnn, minibatch[i], float_type);
#elif defined(TRT)
    for(int i = 0; i < N_DEVICES; i++)
        netModel[nn_id][i] = new TrtModel(pnn, minibatch[i], float_type);
#endif
    delete[] minibatch;

    /*Load NN on each device squentially*/
    strcpy(pnn->path, path);
    Model::dev_type = dev_type;

    for(int dev_id = 0; dev_id < N_DEVICES; dev_id++)
        netModel[nn_id][dev_id]->LoadGraph(dev_id, Model::dev_type);

#if 1
    /*warm up nn*/
    for(int dev_id = 0; dev_id < N_DEVICES; dev_id++) {

        int piece = 0, square = 0;
        Model* net = netModel[nn_id][dev_id];
        for(int i = 0;i < net->BATCH_SIZE;i++)
            add_to_batch(net, 0, 0, 0, 0);
        net->n_batch = 0;

#if 0
        /*time the backend*/
        TIMER s, e;
        get_perf(s);
        for(int i = 0; i < 100; i++)
            net->predict();
        get_perf(e);
        double ms = get_diff(s,e) / 1e8;

        printf("DEV %d bench: %.2f evals/sec\n",
            dev_id, net->BATCH_SIZE * 1e3 / ms);
        fflush(stdout);
#else
        net->predict();
#endif
    }
#endif

    /*Message*/
    printf("Neural network loaded !\t\n");
    fflush(stdout);
}

/*
   Add position to batch
*/

static int add_to_batch(Model* net, float** iplanes, float** p_outputs,
    int* p_size, unsigned short** p_index
    ) {

    int offset = l_add(net->n_batch,1);

    //outputs
    if(p_index) {
        for(int i = 0; i < net->pnn->output_layer_names.size(); i++) {
            net->p_index[i][offset] = p_index[i];
            net->p_size[i][offset] = p_size[i];
            net->p_outputs[i][offset] = p_outputs[i];
        }
    }

    //inputs
    for(int n = 0; n < net->pnn->input_layer_names.size(); n++) {
        float* pinput = net->get_input_buffer(n);
        int sz = net->get_input_size(n);
        pinput += offset * sz;
        if(iplanes)
            memcpy(pinput, iplanes[n], sizeof(float) * sz);
    }

    return offset + 1;
}

/*
   Evaluate position using NN
*/

#define SLEEP() {     \
    t_yield();        \
    t_sleep(delayms); \
}

DLLExport void  _CDECL probe_neural_network(
    float** iplanes, float** p_outputs,
    int* p_size, unsigned short** p_index, 
    uint64_t hash_key, bool hard_probe, int nn_id
    ) {

    //retrieve from cache
    NeuralNet* pnn = netModel[nn_id][0]->pnn;
    if(!hard_probe) {
        if(pnn->retrieve_nn_cache(hash_key,p_index,p_size,p_outputs))
            return;
    }

RETRY:
    //choose GPU device
    Model* net = 0;
    {
        while(true) {
            int device_id, start = chosen_device;
            bool found = false;
            for(int idx = 0; idx < N_DEVICES; idx++) {
                device_id = (start + idx) % N_DEVICES;
                net = netModel[nn_id][device_id];

                if(!net->n_batch_eval && !net->n_finished_threads) {
                    if(l_add(net->n_batch_i, 1) < net->BATCH_SIZE) {
                        if(scheduling == ROUNDROBIN)
                            device_id = (start + idx + 1) % N_DEVICES;
                        l_set(chosen_device, device_id);
                        found = true;
                        break;
                    } else {
                        l_add(net->n_batch_i, -1);
                    }
                }
            }
            if(found) break;

            //sleep
            Model::wait_gpu();
        }
    }

    //add to batch
    int n_thread_batch = add_to_batch(net, iplanes, p_outputs, p_size, p_index);

    //pause threads till eval completes
    if(n_thread_batch < net->BATCH_SIZE) {

        do {

            //a thread reached here after we started evaluating
            if(net->n_batch_eval && (n_thread_batch > net->n_batch_eval)) {
                l_add(net->n_batch, -1);
                l_add(net->n_batch_i, -1);
                goto RETRY;
            }

            //sleep
            SLEEP();

            //this is the last active thread
            if(n_thread_batch == net->n_batch
               && n_active_searchers < n_searchers
               && net->n_batch >= net->BATCH_SIZE - (n_searchers - n_active_searchers)
               ) {
#if 0
                printf("\n[part] # batchsize %d / %d  = active workers %d of %d\n",
                    (int)net->n_batch, net->BATCH_SIZE,
                    (int)n_active_searchers, (int)n_searchers);
                fflush(stdout);
#endif
                l_set(net->n_batch_eval, n_thread_batch);
                l_add(Model::n_idle_gpus, -1);
                net->predict();
                break;
            //sleep
            } else {
                net->wait_eval();
            }

        } while(net->n_finished_threads == 0);

    } else {
#if 0
        printf("\n[full] # batchsize %d / %d  = active workers %d of %d\n",
            (int)net->n_batch, net->BATCH_SIZE,
            (int)n_active_searchers, (int)n_searchers);
        fflush(stdout);
#endif
        net->n_batch_eval = n_thread_batch;
        l_add(Model::n_idle_gpus, -1);
        net->predict();
    }

    //wake up other threads as soon as possible
    int prev_n = l_add(net->n_finished_threads, 1);
    if(prev_n == 0)
        net->signal_eval();

    //store in cache
    net->pnn->store_nn_cache(hash_key,p_index,p_size,p_outputs);

    //last thread to leave resets variables
    ++prev_n;
    if(prev_n == net->n_batch_eval) {
        l_add(net->n_batch, -prev_n);
        l_add(net->n_batch_i, -prev_n);
        net->n_batch_eval = 0;
        net->n_finished_threads = 0;
        l_add(Model::n_idle_gpus, 1);
        Model::signal_gpu();
    }
}

#undef SLEEP

/*
   Set number of active workers
*/
DLLExport void _CDECL set_num_active_searchers(int n_searchers) {
    l_set(n_active_searchers,n_searchers);
}
