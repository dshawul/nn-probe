#include <cstring>
#include <stdio.h>
#include "nnprobe.h"

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
  int device_type = _CPU;
  int n_devices = 1;
  int max_threads = 1;
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
       max_threads, float_type, delay
  );

  return 0;
}
