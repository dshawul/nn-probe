#include <stdio.h>
#include <cstring>
#include <thread>
#include <chrono>
#include "nnprobe.h"

void silly_init(float*, bool);
void thread_proc(int);
static bool cache = false;
static int device_type = _CPU;

int main(int argc, char** argv) {

  /* device type argument */
  if(argc < 2) {
     return 1;
  }

  /* network properties */
  char network[256] = "../../nets/net-2x32.pb";
  char input_names[] = "main_input aux_input";
  char output_names[] = "value/Softmax policy/Reshape";
  char input_shapes[] = "24 8 8   5 1 1";
  char output_sizes[] = "3 256";

  /* loading options */
  int cache_size = 4194304;
  int n_devices = 1;
  int max_threads = 32;
  int float_type = _FLOAT;
  int delay = 0;

  if(!strcmp(argv[1],"-g")) {
     strcpy(network, "../../nets/net-2x32.uff");
     device_type = _GPU;
  }

  /* load the network */
  load_neural_network(
       network,
       input_names, output_names, input_shapes, output_sizes, 
       cache_size, device_type, n_devices,
       max_threads, float_type, delay, 0
  );
 
  /* create max threads */
  std::thread myThreads[max_threads];
  printf("===============================================================\n");  
  printf("Create 32 threads each with a delay of 0.5 seconds for visualizing.\n"
          "Note that evaluation will not begin until all threads call probe.\n");
  printf("===============================================================\n");  
  for(int i = 0; i < max_threads; i++) {
     std::this_thread::sleep_for(std::chrono::milliseconds(500));
     myThreads[i] = std::thread(thread_proc, i);
  }
  for(int i = 0; i < max_threads; i++)
     myThreads[i].join();

  printf("===============================================================\n");  
  printf("Now use only 10 threads and see if it evaluates immediately...\n");
  printf("===============================================================\n");  
  for(int i = 0; i < 10; i++) {
     myThreads[i] = std::thread(thread_proc, i);
     std::this_thread::sleep_for(std::chrono::milliseconds(500));
  }
  printf("===============================================================\n");  
  printf("Waiting for 10 seconds to see if evaluation occurs...\n");
  printf("===============================================================\n");  
  std::this_thread::sleep_for(std::chrono::milliseconds(10000));
  printf("Nope, so forcing evaluation now.\n");
  printf("===============================================================\n");  
  set_num_active_searchers(10);
  for(int i = 0; i < 10; i++)
     myThreads[i].join();
  printf("===============================================================\n");  
  printf("However, cached values could be returned immediately so beware.\n");
  printf("So debug your code with cache turned off first.\n");
  printf("===============================================================\n");  
  cache = true;
  for(int i = 0; i < 10; i++) {
     std::this_thread::sleep_for(std::chrono::milliseconds(500));
     myThreads[i] = std::thread(thread_proc, i);
  }
  for(int i = 0; i < 10; i++)
     myThreads[i].join();
  printf("===============================================================\n");  
  printf("Finished multi-threaded test.\n");
  printf("===============================================================\n");  

  return 0;
}

void thread_proc(int id) {

  /* buffers = initialized to 0 */
  float* main_input = new float[24 * 8 * 8];
  float* aux_input = new float[5 * 1 * 1];
  float* value_head = new float[3];
  float* policy_head = new float[256];
  unsigned short* policy_index = new unsigned short[256];

  /*... initialize inputs in somewhat better way ...*/
  silly_init(main_input, (device_type == _GPU));
  memset(aux_input, 0, 5 * sizeof(float));
  for(int i = 0; i < 128; i++)
     policy_index[i] = i;

  /* get all of value_head */
  unsigned short* out_index[2] = {0, policy_index};

  int out_size[2] = {3, 128};
  float* inputs[2] = {main_input, aux_input};
  float* outputs[2] = {value_head,policy_head};
  uint64_t hkey = (uint64_t)(0x44c3964f82358793);
  bool hard_probe = !cache;

  printf("[Thread %2d] probing neural.\n", id );

  probe_neural_network(
      inputs, outputs,
      out_size, out_index,
      hkey, hard_probe, 0
  );

  float p = value_head[0] * 1.0 + value_head[1] * 0.5;
  
  printf("[Thread %2d] Winning probability %.6f\n", id, p );
}

void silly_init(float* main_input, bool NCHW) {

  static float temp[] = {
      //
      // First 12 channels are attacks by pieces
      //

      //Channel 0 = wking
      0, 0, 0, 1, 0, 1, 0, 0, 
      0, 0, 0, 1, 1, 1, 0, 0, 
      0, 0, 0, 0, 0, 0, 0, 0, 
      0, 0, 0, 0, 0, 0, 0, 0, 
      0, 0, 0, 0, 0, 0, 0, 0, 
      0, 0, 0, 0, 0, 0, 0, 0, 
      0, 0, 0, 0, 0, 0, 0, 0, 
      0, 0, 0, 0, 0, 0, 0, 0, 

      //Channel 1 = wqueen
      0, 0, 1, 0, 1, 0, 0, 0, 
      0, 0, 1, 1, 1, 0, 0, 0, 
      0, 0, 0, 0, 0, 0, 0, 0, 
      0, 0, 0, 0, 0, 0, 0, 0, 
      0, 0, 0, 0, 0, 0, 0, 0, 
      0, 0, 0, 0, 0, 0, 0, 0, 
      0, 0, 0, 0, 0, 0, 0, 0, 
      0, 0, 0, 0, 0, 0, 0, 0, 

      //Channel 2 = wrook
      0, 1, 0, 0, 0, 0, 1, 0, 
      1, 0, 0, 0, 0, 0, 0, 1, 
      0, 0, 0, 0, 0, 0, 0, 0, 
      0, 0, 0, 0, 0, 0, 0, 0, 
      0, 0, 0, 0, 0, 0, 0, 0, 
      0, 0, 0, 0, 0, 0, 0, 0, 
      0, 0, 0, 0, 0, 0, 0, 0, 
      0, 0, 0, 0, 0, 0, 0, 0, 

      //Channel 3 = wbishop
      0, 0, 0, 0, 0, 0, 0, 0, 
      0, 1, 0, 1, 1, 0, 1, 0, 
      0, 0, 0, 0, 0, 0, 0, 0, 
      0, 0, 0, 0, 0, 0, 0, 0, 
      0, 0, 0, 0, 0, 0, 0, 0, 
      0, 0, 0, 0, 0, 0, 0, 0, 
      0, 0, 0, 0, 0, 0, 0, 0, 
      0, 0, 0, 0, 0, 0, 0, 0, 

      //Channel 4 = wknight
      0, 0, 0, 0, 0, 0, 0, 0, 
      0, 0, 0, 1, 1, 0, 0, 0, 
      1, 0, 1, 0, 0, 1, 0, 1, 
      0, 0, 0, 0, 0, 0, 0, 0, 
      0, 0, 0, 0, 0, 0, 0, 0, 
      0, 0, 0, 0, 0, 0, 0, 0, 
      0, 0, 0, 0, 0, 0, 0, 0, 
      0, 0, 0, 0, 0, 0, 0, 0, 

      //Channel 5 = wpawns
      0, 0, 0, 0, 0, 0, 0, 0, 
      0, 0, 0, 0, 0, 0, 0, 0, 
      1, 1, 1, 1, 1, 1, 1, 1, 
      0, 0, 0, 0, 0, 0, 0, 0, 
      0, 0, 0, 0, 0, 0, 0, 0, 
      0, 0, 0, 0, 0, 0, 0, 0, 
      0, 0, 0, 0, 0, 0, 0, 0, 
      0, 0, 0, 0, 0, 0, 0, 0, 

      //Channel 6 = bking
      0, 0, 0, 0, 0, 0, 0, 0, 
      0, 0, 0, 0, 0, 0, 0, 0, 
      0, 0, 0, 0, 0, 0, 0, 0, 
      0, 0, 0, 0, 0, 0, 0, 0, 
      0, 0, 0, 0, 0, 0, 0, 0, 
      0, 0, 0, 0, 0, 0, 0, 0, 
      0, 0, 0, 1, 1, 1, 0, 0, 
      0, 0, 0, 1, 0, 1, 0, 0, 

      //Channel 7 = bqueen
      0, 0, 0, 0, 0, 0, 0, 0, 
      0, 0, 0, 0, 0, 0, 0, 0, 
      0, 0, 0, 0, 0, 0, 0, 0, 
      0, 0, 0, 0, 0, 0, 0, 0, 
      0, 0, 0, 0, 0, 0, 0, 0, 
      0, 0, 0, 0, 0, 0, 0, 0, 
      0, 0, 1, 1, 1, 0, 0, 0, 
      0, 0, 1, 0, 1, 0, 0, 0, 

      //Channel 8 = brook
      0, 0, 0, 0, 0, 0, 0, 0, 
      0, 0, 0, 0, 0, 0, 0, 0, 
      0, 0, 0, 0, 0, 0, 0, 0, 
      0, 0, 0, 0, 0, 0, 0, 0, 
      0, 0, 0, 0, 0, 0, 0, 0, 
      0, 0, 0, 0, 0, 0, 0, 0, 
      1, 0, 0, 0, 0, 0, 0, 1, 
      0, 1, 0, 0, 0, 0, 1, 0, 

      //Channel 9 = bbishop
      0, 0, 0, 0, 0, 0, 0, 0, 
      0, 0, 0, 0, 0, 0, 0, 0, 
      0, 0, 0, 0, 0, 0, 0, 0, 
      0, 0, 0, 0, 0, 0, 0, 0, 
      0, 0, 0, 0, 0, 0, 0, 0, 
      0, 0, 0, 0, 0, 0, 0, 0, 
      0, 1, 0, 1, 1, 0, 1, 0, 
      0, 0, 0, 0, 0, 0, 0, 0, 

      //Channel 10 = bknight
      0, 0, 0, 0, 0, 0, 0, 0, 
      0, 0, 0, 0, 0, 0, 0, 0, 
      0, 0, 0, 0, 0, 0, 0, 0, 
      0, 0, 0, 0, 0, 0, 0, 0, 
      0, 0, 0, 0, 0, 0, 0, 0, 
      1, 0, 1, 0, 0, 1, 0, 1, 
      0, 0, 0, 1, 1, 0, 0, 0, 
      0, 0, 0, 0, 0, 0, 0, 0, 

      //Channel 11 = bpawn
      0, 0, 0, 0, 0, 0, 0, 0, 
      0, 0, 0, 0, 0, 0, 0, 0, 
      0, 0, 0, 0, 0, 0, 0, 0, 
      0, 0, 0, 0, 0, 0, 0, 0, 
      0, 0, 0, 0, 0, 0, 0, 0, 
      1, 1, 1, 1, 1, 1, 1, 1, 
      0, 0, 0, 0, 0, 0, 0, 0, 
      0, 0, 0, 0, 0, 0, 0, 0, 

      //
      // Last 12 channels are piece placments
      // in the same order as above
      //

      //Channel 12
      0, 0, 0, 0, 1, 0, 0, 0, 
      0, 0, 0, 0, 0, 0, 0, 0, 
      0, 0, 0, 0, 0, 0, 0, 0, 
      0, 0, 0, 0, 0, 0, 0, 0, 
      0, 0, 0, 0, 0, 0, 0, 0, 
      0, 0, 0, 0, 0, 0, 0, 0, 
      0, 0, 0, 0, 0, 0, 0, 0, 
      0, 0, 0, 0, 0, 0, 0, 0, 

      //Channel 13
      0, 0, 0, 1, 0, 0, 0, 0, 
      0, 0, 0, 0, 0, 0, 0, 0, 
      0, 0, 0, 0, 0, 0, 0, 0, 
      0, 0, 0, 0, 0, 0, 0, 0, 
      0, 0, 0, 0, 0, 0, 0, 0, 
      0, 0, 0, 0, 0, 0, 0, 0, 
      0, 0, 0, 0, 0, 0, 0, 0, 
      0, 0, 0, 0, 0, 0, 0, 0, 

      //Channel 14
      1, 0, 0, 0, 0, 0, 0, 1, 
      0, 0, 0, 0, 0, 0, 0, 0, 
      0, 0, 0, 0, 0, 0, 0, 0, 
      0, 0, 0, 0, 0, 0, 0, 0, 
      0, 0, 0, 0, 0, 0, 0, 0, 
      0, 0, 0, 0, 0, 0, 0, 0, 
      0, 0, 0, 0, 0, 0, 0, 0, 
      0, 0, 0, 0, 0, 0, 0, 0, 

      //Channel 15
      0, 0, 1, 0, 0, 1, 0, 0, 
      0, 0, 0, 0, 0, 0, 0, 0, 
      0, 0, 0, 0, 0, 0, 0, 0, 
      0, 0, 0, 0, 0, 0, 0, 0, 
      0, 0, 0, 0, 0, 0, 0, 0, 
      0, 0, 0, 0, 0, 0, 0, 0, 
      0, 0, 0, 0, 0, 0, 0, 0, 
      0, 0, 0, 0, 0, 0, 0, 0, 

      //Channel 16
      0, 1, 0, 0, 0, 0, 1, 0, 
      0, 0, 0, 0, 0, 0, 0, 0, 
      0, 0, 0, 0, 0, 0, 0, 0, 
      0, 0, 0, 0, 0, 0, 0, 0, 
      0, 0, 0, 0, 0, 0, 0, 0, 
      0, 0, 0, 0, 0, 0, 0, 0, 
      0, 0, 0, 0, 0, 0, 0, 0, 
      0, 0, 0, 0, 0, 0, 0, 0, 

      //Channel 17
      0, 0, 0, 0, 0, 0, 0, 0, 
      1, 1, 1, 1, 1, 1, 1, 1, 
      0, 0, 0, 0, 0, 0, 0, 0, 
      0, 0, 0, 0, 0, 0, 0, 0, 
      0, 0, 0, 0, 0, 0, 0, 0, 
      0, 0, 0, 0, 0, 0, 0, 0, 
      0, 0, 0, 0, 0, 0, 0, 0, 
      0, 0, 0, 0, 0, 0, 0, 0, 

      //Channel 18
      0, 0, 0, 0, 0, 0, 0, 0, 
      0, 0, 0, 0, 0, 0, 0, 0, 
      0, 0, 0, 0, 0, 0, 0, 0, 
      0, 0, 0, 0, 0, 0, 0, 0, 
      0, 0, 0, 0, 0, 0, 0, 0, 
      0, 0, 0, 0, 0, 0, 0, 0, 
      0, 0, 0, 0, 0, 0, 0, 0, 
      0, 0, 0, 0, 1, 0, 0, 0, 

      //Channel 19
      0, 0, 0, 0, 0, 0, 0, 0, 
      0, 0, 0, 0, 0, 0, 0, 0, 
      0, 0, 0, 0, 0, 0, 0, 0, 
      0, 0, 0, 0, 0, 0, 0, 0, 
      0, 0, 0, 0, 0, 0, 0, 0, 
      0, 0, 0, 0, 0, 0, 0, 0, 
      0, 0, 0, 0, 0, 0, 0, 0, 
      0, 0, 0, 1, 0, 0, 0, 0, 

      //Channel 20
      0, 0, 0, 0, 0, 0, 0, 0, 
      0, 0, 0, 0, 0, 0, 0, 0, 
      0, 0, 0, 0, 0, 0, 0, 0, 
      0, 0, 0, 0, 0, 0, 0, 0, 
      0, 0, 0, 0, 0, 0, 0, 0, 
      0, 0, 0, 0, 0, 0, 0, 0, 
      0, 0, 0, 0, 0, 0, 0, 0, 
      1, 0, 0, 0, 0, 0, 0, 1, 

      //Channel 21
      0, 0, 0, 0, 0, 0, 0, 0, 
      0, 0, 0, 0, 0, 0, 0, 0, 
      0, 0, 0, 0, 0, 0, 0, 0, 
      0, 0, 0, 0, 0, 0, 0, 0, 
      0, 0, 0, 0, 0, 0, 0, 0, 
      0, 0, 0, 0, 0, 0, 0, 0, 
      0, 0, 0, 0, 0, 0, 0, 0, 
      0, 0, 1, 0, 0, 1, 0, 0, 

      //Channel 22
      0, 0, 0, 0, 0, 0, 0, 0, 
      0, 0, 0, 0, 0, 0, 0, 0, 
      0, 0, 0, 0, 0, 0, 0, 0, 
      0, 0, 0, 0, 0, 0, 0, 0, 
      0, 0, 0, 0, 0, 0, 0, 0, 
      0, 0, 0, 0, 0, 0, 0, 0, 
      0, 0, 0, 0, 0, 0, 0, 0, 
      0, 1, 0, 0, 0, 0, 1, 0, 

      //Channel 23
      0, 0, 0, 0, 0, 0, 0, 0, 
      0, 0, 0, 0, 0, 0, 0, 0, 
      0, 0, 0, 0, 0, 0, 0, 0, 
      0, 0, 0, 0, 0, 0, 0, 0, 
      0, 0, 0, 0, 0, 0, 0, 0, 
      0, 0, 0, 0, 0, 0, 0, 0, 
      1, 1, 1, 1, 1, 1, 1, 1, 
      0, 0, 0, 0, 0, 0, 0, 0
  };

  //channels-first on gpu
  if(NCHW) {
      memcpy(main_input, temp, 24 * 8 * 8 * sizeof(float));
  //channels-last on cpu
  } else {
      for(int c = 0; c < 24; c++) {
          for(int j = 0; j < 8; j++) {
              for(int i = 0; i < 8; i++) {
                  main_input[j * 8 * 24 + i * 24 + c] = temp[c * 8 * 8 + j * 8 + i];
              }
          }
      }
  }

}
