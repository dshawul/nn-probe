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
static int BATCH_SIZE;
static int n_searchers;
static VOLATILE int n_active_searchers;
static VOLATILE int chosen_device = 0;
static int delayms = 0;
static int floatPrecision = 1;

static LOCK global_lock;

/*
Neural network properties
*/
struct NeuralNet {
    std::vector<std::string> input_layer_names;
    std::vector<std::string> output_layer_names;
    std::vector<std::tuple<int,int,int>> input_layer_shapes;
    std::vector<int> output_layer_sizes;
    char path[256];

    UBMP64* nn_cache;
    UBMP32 nn_cache_mask;
    UBMP32 hash_entry_sz;

    void allocate_nn_cache(UBMP32 sizeb);
    void store_nn_cache(const UBMP64 hash_key,  unsigned short** const p_index,
                           int* const p_size, float** const p_outputs);
    bool retrieve_nn_cache(const UBMP64 hash_key, unsigned short** const p_index,
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
    VOLATILE int wait;
    VOLATILE int n_batch;
    VOLATILE int n_batch_i;
    VOLATILE int n_finished_threads;
    int id;
    Model(NeuralNet* pnn_) {
        pnn = pnn_;
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
        n_finished_threads = 0;
        id = 0;
        wait = 1;
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
public:
    TfModel(NeuralNet*);
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

TfModel::TfModel(NeuralNet* pnn_) : Model(pnn_) {
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
                printf("%d. %s = ", n, node.name().c_str());
                for (int i = 1; i < shape.dim_size(); i++) {
                    printf("%d ",(int)shape.dim(i).size());
                    nshape.AddDim(shape.dim(i).size());
                }
                printf("\n");
                input_layers[n] = new Tensor(DT_FLOAT, nshape);
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
}

void TfModel::predict() {
    std::vector<Tensor> outputs;
    std::vector<std::pair<std::string, Tensor> > inps;
    std::vector<std::string> outs;

    for(int n = 0; n < pnn->input_layer_names.size(); n++) {
        std::pair<std::string, Tensor> pr( 
            pnn->input_layer_names[n], *(input_layers[n]) );
        inps.push_back(pr);
    }
    for(int n = 0; n < pnn->output_layer_names.size(); n++)
        outs.push_back(pnn->output_layer_names[n]);

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
  NeuralNet* pnn;
public:

  Int8CacheCalibrator(NeuralNet* pnn_) {
    pnn = pnn_;

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

    if (floatPrecision == 2) {
        epd_file = fopen(calib_file_name.c_str(),"rb");
        if(!epd_file) {
            printf("Calibration file not found!\n");
            fflush(stdout);
            exit(0);
        }
    }
  }

  ~Int8CacheCalibrator() override {
    for(int n = 0; n < pnn->input_layer_names.size(); n++) {
        cudaFree(buffers[n]);
        free(buffers_h[n]);
    }
    if(epd_file)
        fclose(epd_file);
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
    nvinfer1::DataType floatMode;
    std::vector<void*> buffers;
    std::vector<float*> buffers_h;
    std::vector<int> buffer_sizes;
    std::vector<int> inp_index;
    std::vector<int> out_index;
public:
    TrtModel(NeuralNet*);
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

TrtModel::TrtModel(NeuralNet* pnn_) : Model(pnn_) {
    context = 0;
    engine = 0;
    numBindings = 0;
    if(floatPrecision == 0)
        floatMode = nvinfer1::DataType::kFLOAT;
    else if(floatPrecision == 1)
        floatMode = nvinfer1::DataType::kHALF;
    else
        floatMode = nvinfer1::DataType::kINT8;
}

TrtModel::~TrtModel() {
    context->destroy();
    engine->destroy();
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
                          std::to_string(floatPrecision) +
                          ".trt";
    std::ifstream ifs(trtName.c_str(), std::ios::in | std::ios::binary);

    /*wait until gpu-0 writes trt file*/
    static VOLATILE int is_ready = 0;
    if(!ifs.is_open()) {
        if(Model::id == 0)
            is_ready = 0;
        else  {
            while(!is_ready)
                t_sleep(10);
            ifs.open(trtName.c_str(), std::ios::in | std::ios::binary);
        }
    }

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

        IBuilder* builder = createInferBuilder(logger);
        if ((floatMode == nvinfer1::DataType::kINT8 && !builder->platformHasFastInt8()) 
         || (floatMode == nvinfer1::DataType::kHALF && !builder->platformHasFastFp16())) {
            std::cout << "Device does not support this low precision mode." << std::endl;
            return;
        }

        INetworkDefinition* network = builder->createNetwork();
        nvinfer1::DataType loadMode = (floatMode == nvinfer1::DataType::kINT8) ?
                            nvinfer1::DataType::kFLOAT : floatMode;
        if(!parser->parse(pnn->path, *network, loadMode)) {
            std::cout << "Fail to parse network " << pnn->path << std::endl;
            return;
        }

        Int8CacheCalibrator calibrator(pnn);

        if (floatMode == nvinfer1::DataType::kHALF) {
            builder->setFp16Mode(true);
        } else if (floatMode == nvinfer1::DataType::kINT8) {
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

        network->destroy();
        builder->destroy();
        parser->destroy();

        if(Model::id == 0) {
            IHostMemory *trtModelStream = engine->serialize();
            FILE* out_file = fopen(trtName.c_str(),"wb");
            fwrite((char*)(trtModelStream->data()), 1, trtModelStream->size(), out_file);
            fclose(out_file);
            trtModelStream->destroy();
            is_ready = 1;
        }
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
        if(dev_id == 0) {
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

void NeuralNet::allocate_nn_cache(UBMP32 sizeb) {
    hash_entry_sz = sizeof(UBMP64);
    for(int k = 0; k < output_layer_sizes.size(); k++) {
        hash_entry_sz += output_layer_sizes[k] * 
                (sizeof(unsigned short) + sizeof(float));
    }
    hash_entry_sz = sizeof(UBMP64) * ( (hash_entry_sz + sizeof(UBMP64) - 1) / sizeof(UBMP64));

    UBMP32 size = 1, size_max = sizeb / hash_entry_sz;
    while(2 * size <= size_max) size *= 2;
    nn_cache_mask = size - 1;
    hash_entry_sz /= sizeof(UBMP64);
    aligned_reserve<UBMP64>( nn_cache, size * hash_entry_sz );

    printf("nn_cache %d X %d = %.1f MB\n",size,int(hash_entry_sz * sizeof(UBMP64)),
        (size * hash_entry_sz * sizeof(UBMP64)) / double(1024 * 1024));
    fflush(stdout);
}

void NeuralNet::store_nn_cache(const UBMP64 hash_key,  unsigned short** const p_index,
                           int* const p_size, float** const p_outputs
    ) {
    UBMP32 key = UBMP32(hash_key & nn_cache_mask);
    UBMP64* const nn_hash = nn_cache + key * hash_entry_sz; 
    
    if(*nn_hash != hash_key) {
        *nn_hash = hash_key;
        UBMP16* p = (UBMP16*) (nn_hash + 1);
        for(int k = 0; k < output_layer_names.size(); k++) {
            memcpy(p, p_outputs[k], p_size[k] * sizeof(float));
            p += p_size[k] * 2;
            if(p_index[k]) {
                memcpy(p, p_index[k], p_size[k] * sizeof(UBMP16));
                p += p_size[k];
            }
        }
    }
}

bool NeuralNet::retrieve_nn_cache(const UBMP64 hash_key, unsigned short** const p_index,
                              int* const p_size, float** p_outputs
    ) {
    UBMP32 key = UBMP32(hash_key & nn_cache_mask);
    UBMP64* const nn_hash = nn_cache + key * hash_entry_sz;

    if(*nn_hash == hash_key) {
        UBMP16* p = (UBMP16*) (nn_hash + 1);
        for(int k = 0; k < output_layer_names.size(); k++) {
            if(p_index[k]) {
                float* const nn_outputs = (float*)p;
                p += p_size[k] * 2;
                UBMP16* const nn_index = (UBMP16*)(p);
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
  Thread procedure for loading NN
*/
static VOLATILE int nn_loaded = 0;
static int nn_id_global = 0;
static void CDECL nn_thread_proc(void* id) {
    int dev_id = *((int*)id);
    netModel[nn_id_global][dev_id]->LoadGraph(dev_id, Model::dev_type);
    l_add(nn_loaded,1);
}

/*
   Initialize tensorflow
*/
static int add_to_batch(int nn_id, int batch_id, float** iplanes);

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
    int max_threads, int float_type, int delay, int nn_id
    ) {

    l_create(global_lock);

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
    BATCH_SIZE = n_searchers / N_DEVICES;
    floatPrecision = float_type;

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
    static const char* float_type_string[] = {"FLOAT", "HALF", "INT8"};
    printf("Loading neural network : %s\n",path);
    printf("With \"%s\" precision\n", float_type_string[floatPrecision]);
    fflush(stdout);

    /*Load tensorflow or tensorrt graphs on GPU*/
    netModel[nn_id] = new Model*[N_DEVICES];
#if defined(TENSORFLOW) && defined(TRT)
    if(strstr(path, ".pb") != NULL) {
        for(int i = 0; i < N_DEVICES; i++)
            netModel[nn_id][i] = new TfModel(pnn);
    } else if(strstr(path, ".uff") != NULL) {
        for(int i = 0; i < N_DEVICES; i++)
            netModel[nn_id][i] = new TrtModel(pnn);
    }
#elif defined(TENSORFLOW)
    for(int i = 0; i < N_DEVICES; i++)
        netModel[nn_id][i] = new TfModel(pnn);
#elif defined(TRT)
    for(int i = 0; i < N_DEVICES; i++)
        netModel[nn_id][i] = new TrtModel(pnn);
#endif

    /*Load NN with multiple threads*/
    strcpy(pnn->path, path);
    Model::dev_type = dev_type;
    nn_id_global = nn_id;
    int* tid = new int[N_DEVICES];
    for(int dev_id = 0; dev_id < N_DEVICES; dev_id++) {
        tid[dev_id] = dev_id;
        pthread_t dummy;
        t_create(dummy,nn_thread_proc,&tid[dev_id]);
        (void)dummy;
    }
    while(nn_loaded < N_DEVICES)
        t_sleep(1);
    delete[] tid;
#if 1
    /*warm up nn*/
    for(int dev_id = 0; dev_id < N_DEVICES; dev_id++) {
        int piece = 0, square = 0;
        Model* net = netModel[nn_id][dev_id];
        for(int i = 0;i < BATCH_SIZE;i++)
            add_to_batch(nn_id, dev_id, 0);
        net->n_batch = 0;
        net->predict();
    }
#endif
    nn_loaded = 0;
    /*Message*/
    printf("Neural network loaded !\t\n");
    fflush(stdout);
}

/*
   Add position to batch
*/

static int add_to_batch(int nn_id, int batch_id, float** iplanes) {

    //get identifier
    Model* net = netModel[nn_id][batch_id];

    //offsets
    int offset = l_add(net->n_batch,1);
    for(int n = 0; n < net->pnn->input_layer_names.size(); n++) {
        float* pinput = net->get_input_buffer(n);
        int sz = net->get_input_size(n);
        pinput += offset * sz;
        if(iplanes)
            memcpy(pinput, iplanes[n], sizeof(float) * sz);
    }

    return offset;
}

/*
   Evaluate position using NN
*/

#define SLEEP() {     \
    t_yield();        \
    t_sleep(delayms); \
}

DLLExport void  CDECL probe_neural_network(
    float** iplanes, float** p_outputs,
    int* p_size, unsigned short** p_index, 
    UBMP64 hash_key, bool hard_probe, int nn_id
    ) {

    //retrieve from cache
    NeuralNet* pnn = netModel[nn_id][0]->pnn;
    if(!hard_probe) {
        if(pnn->retrieve_nn_cache(hash_key,p_index,p_size,p_outputs))
            return;
    }

    //choose batch id
    int batch_id;
    l_lock(global_lock);
    do {
        for(batch_id = chosen_device;batch_id < N_DEVICES; batch_id++) {
            Model* net = netModel[nn_id][batch_id];
            if(net->n_batch_i < BATCH_SIZE) {
                net->n_batch_i++;
                break;
            }
        }
        chosen_device = 0;
    } while(batch_id + 1 > N_DEVICES);

    chosen_device = batch_id + 1;
    if(chosen_device == N_DEVICES)
        chosen_device = 0;
    l_unlock(global_lock);

    //get identifier
    Model* net = netModel[nn_id][batch_id];

    //add to batch
    int offset = add_to_batch(nn_id, batch_id, iplanes);

    //outputs
    for(int i = 0; i < net->pnn->output_layer_names.size(); i++) {
        net->p_index[i][offset] = p_index[i];
        net->p_size[i][offset] = p_size[i];
        net->p_outputs[i][offset] = p_outputs[i];
    }

    //pause threads till eval completes
    if(offset + 1 < BATCH_SIZE) {

        while(net->wait) {
            SLEEP();

            if(offset + 1 == net->n_batch
               && n_active_searchers < n_searchers 
               && net->n_batch >= BATCH_SIZE - (n_searchers - n_active_searchers)
               ) {
#if 0
                    printf("\n# batchsize %d / %d workers %d / %d\n",
                        net->n_batch,BATCH_SIZE,
                        n_active_searchers, n_searchers);
                    fflush(stdout);
#endif
                net->predict();
                net->wait = 0;
                break;
            }
        }

    } else {
        net->predict();
        net->wait = 0;
    }

    //store in cache
    net->pnn->store_nn_cache(hash_key,p_index,p_size,p_outputs);

    //Wait until all eval calls are finished
    l_lock(global_lock);
    net->n_finished_threads++;
    if(net->n_finished_threads == net->n_batch) {
        net->wait = 1;
        net->n_finished_threads = 0;
        net->n_batch = 0;
        net->n_batch_i = 0;
    } 
    l_unlock(global_lock);

    while (net->n_finished_threads > 0 
        && net->n_finished_threads < net->n_batch
        ) {
        SLEEP();
    }
}

#undef SLEEP

/*
   Set number of active workers
*/
DLLExport void CDECL set_num_active_searchers(int n_searchers) {
    l_set(n_active_searchers,n_searchers);
}