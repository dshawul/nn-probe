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
#include "include/cuda_runtime_api.h"
#include "include/NvInfer.h"
#include "include/NvUffParser.h"
#endif

#define DLL_EXPORT
#include "nnprobe.h"
#undef DLL_EXPORT

#include "my_types.h"

static int N_DEVICES;
static int n_searchers;
static VOLATILE int n_active_searchers;
static VOLATILE int chosen_device = 0;
static int delayms = 0;
static int batch_size_factor = 0;

enum { FCFS, ROUNDROBIN };
static int scheduling = FCFS;

enum { FP32 = 0, FP16, FP8 };
static const char* float_type_string[] = {"FLOAT", "HALF", "INT8"};

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
    VOLATILE int n_batch;
    VOLATILE int n_batch_i;
    VOLATILE int n_batch_eval;
    VOLATILE int n_finished_threads;
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
};
int Model::dev_type;

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

using namespace nvuffparser;
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

    int getBatchSize() const override {
        return CAL_BATCH_SIZE;
    }

    bool getBatch(void* bindings[], const char* names[], int nbBindings) override {
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

    const void* readCalibrationCache(size_t& length) override {
        return nullptr;
    }

    void writeCalibrationCache(const void* cache, size_t length) override {
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
    void log(Severity severity, const char* msg) override {
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

        IUffParser* parser;
        parser = createUffParser();

        for(int n = 0; n < pnn->input_layer_names.size(); n++)
            parser->registerInput(pnn->input_layer_names[n].c_str(), 
                nvinfer1::DimsCHW(std::get<0>(pnn->input_layer_shapes[n]),
                                  std::get<1>(pnn->input_layer_shapes[n]),
                                  std::get<2>(pnn->input_layer_shapes[n])), 
                                  UffInputOrder::kNCHW);

        for(int n = 0; n < pnn->output_layer_names.size(); n++)
            parser->registerOutput(pnn->output_layer_names[n].c_str());

        /*if requested precision is not supported, fallback to next available*/
        IBuilder* builder = createInferBuilder(logger);
        if(float_type == FP8 && !builder->platformHasFastInt8()) {
            if(builder->platformHasFastFp16()) {
                float_type = FP16;
            } else {
                float_type = FP32;
            }
            printf("Switching to \"%s\" precision for GPU %d\n",
                float_type_string[float_type], dev_id);
            fflush(stdout);
        }
        if(float_type == FP16 && !builder->platformHasFastFp16()) {
            if(builder->platformHasFastInt8()) {
                float_type = FP8;
            } else {
                float_type = FP32;
            }
            printf("Switching to \"%s\" precision for GPU %d\n",
                float_type_string[float_type], dev_id);
            fflush(stdout);
        }
        
        /*create network*/
        INetworkDefinition* network = builder->createNetwork();
        nvinfer1::DataType loadMode;
        if(float_type == FP16)
            loadMode = nvinfer1::DataType::kHALF;
        else
            loadMode = nvinfer1::DataType::kFLOAT;
        if(!parser->parse(pnn->path, *network, loadMode)) {
            std::cout << "Fail to parse network " << pnn->path << std::endl;
            return;
        }

        /*create engine*/
        Int8CacheCalibrator calibrator(pnn, float_type);

        if (float_type == FP16) {
            builder->setFp16Mode(true);
        } else if (float_type == FP8) {
            builder->setInt8Mode(true);
            builder->setInt8Calibrator(&calibrator);
        }
        builder->setMaxBatchSize(BATCH_SIZE);
        builder->setMaxWorkspaceSize((1 << 30));

        engine = builder->buildCudaEngine(*network);
        if (!engine) {
            std::cout << "Unable to create engine" << std::endl;
            return;
        }

        IHostMemory *trtModelStream = engine->serialize();
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
        engine = infer->deserializeCudaEngine(trtModelStream, size, nullptr);
        if (trtModelStream) delete[] trtModelStream;
    }

    context = engine->createExecutionContext();
    numBindings = engine->getNbBindings();
    
    /*Pinned memory*/
    size_t TOTAL = 0;
    for(int i = 0; i < numBindings; i++) {

        Dims d = engine->getBindingDimensions(i);
        size_t size = 1;
        for(size_t j = 0; j < d.nbDims; j++) 
            size*= d.d[j];
        TOTAL += size;
        buffer_sizes.push_back(size);

#if 1
        if(dev_id == N_DEVICES -1) {
            printf("%d. %s %d =",i,engine->getBindingName(i),(int)size);
            for(size_t j = 0; j < d.nbDims; j++) 
                printf(" %d",d.d[j]);
            printf("\n");
            fflush(stdout);
        }
#endif
    }

    float* pDevice, *pHost;
    cudaHostAlloc((void**)&pHost, 
        BATCH_SIZE * TOTAL * sizeof(float), 
        cudaHostAllocMapped);
    cudaHostGetDevicePointer((void**)&pDevice,(void*)pHost,0);

    for(int i = 0; i < numBindings; i++) {
        size_t size = buffer_sizes[i];
        buffers.push_back(pDevice);
        buffers_h.push_back(pHost);
        pDevice += BATCH_SIZE * size;
        pHost += BATCH_SIZE * size;
    }

    for(int n = 0; n < pnn->input_layer_names.size(); n++)
        inp_index.push_back( 
            engine->getBindingIndex(pnn->input_layer_names[n].c_str()) );
    for(int n = 0; n < pnn->output_layer_names.size(); n++)
        out_index.push_back( 
            engine->getBindingIndex(pnn->output_layer_names[n].c_str()) );
}
void TrtModel::predict() {

    cudaSetDevice(Model::id);

    context->execute(BATCH_SIZE, buffers.data());

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
    } else if(strstr(path, ".uff") != NULL) {
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

        printf("DEV %d :  %.2f ms %.2f nps\n",
            dev_id, ms, net->BATCH_SIZE * 1e3 / ms);
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
        int device_id = chosen_device;
        while(true) {
            net = netModel[nn_id][device_id];

            if(!net->n_batch_eval && !net->n_finished_threads) {
                if(l_add(net->n_batch_i, 1) < net->BATCH_SIZE) {
                    if(scheduling == ROUNDROBIN) {
                        device_id++;
                        if(device_id == N_DEVICES)
                            device_id = 0;
                    }
                    l_set(chosen_device, device_id);
                    break;
                } else {
                    l_add(net->n_batch_i, -1);
                }
            }

            device_id++;
            if(device_id == N_DEVICES)
                device_id = 0;
            SLEEP();
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
                    printf("\n[0] # batchsize %d / %d  = active workers %d of %d\n",
                        net->n_batch, net->BATCH_SIZE,
                        n_active_searchers, n_searchers);
                    fflush(stdout);
#endif
                net->n_batch_eval = n_thread_batch;
                net->predict();
                break;
            }

        } while(net->n_finished_threads == 0);

    } else {
        net->n_batch_eval = n_thread_batch;
        net->predict();
    }

    //last thread to leave resets variables
    int prev_n = l_add(net->n_finished_threads, 1) + 1;
    if(prev_n == net->n_batch_eval) {
        l_add(net->n_batch, -prev_n);
        l_add(net->n_batch_i, -prev_n);
        net->n_batch_eval = 0;
        net->n_finished_threads = 0;
    }

    //store in cache
    net->pnn->store_nn_cache(hash_key,p_index,p_size,p_outputs);
}

#undef SLEEP

/*
   Set number of active workers
*/
DLLExport void _CDECL set_num_active_searchers(int n_searchers) {
    l_set(n_active_searchers,n_searchers);
}
