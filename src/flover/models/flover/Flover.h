#include "3rdparty/INIReader.h"
#include "src/flover/utils/nccl_utils.h"
#include "src/flover/utils/nvtx_utils.h"
#include "src/flover/utils/word_list.h"

#include "src/flover/models/gptj/GptJWeight.h"
#include "src/flover/models/llama/LlamaWeight.h"

#include "src/flover/models/gptj/gptj_init.h"
#include "src/flover/models/llama/llama_init.h"

#include "src/flover/models/flover/buffers/gptj_buffers.h"
#include "src/flover/models/flover/buffers/llama_buffers.h"

#include "src/flover/utils/request_queue.h"
#include "src/flover/utils/inference_queue.h"

#include <condition_variable>
#include <thread>
#include <map>
#include <string>
#include <future>
#include <atomic>


namespace flover {

template<typename T>
class Flover {
public:
  Flover (const INIReader& _reader, IAllocator* _allocator) {
    reader = _reader;
    flover_allocator = _allocator;
    initialize();
  }
  void freeBuffer();
  void addRequest(int defined_generation_len);
  void serve(int* stop_flag, int setDeviceNo, std::promise<void> &p_fetching, std::promise<void> &p_inference);
  void test();

  std::vector<std::thread> TRs_FetchingAndPreprocessing;
  bool new_req = false;
  bool added   = false;

protected:
  std::string flover_model_name;
  void initialize();
  void init_cublas();
  void FetchingAndPreprocessing(int* stop_flag, int setDeviceNo, std::promise<void> &p);
  void func_preprocessing(int                                            my_id, 
                          cudaStream_t                                   stream, 
                          NcclParam                                      tensor_para, 
                          NcclParam                                      pipeline_para, 
                          cublasMMWrapper*                               cublas_wrapper,
                          std::unordered_map<std::string, Tensor>*       output_tensors,
                          const std::unordered_map<std::string, Tensor>* input_tensors);
  void InferenceStream(int* stop_flag, int setDeviceNo, std::promise<void> &p);
  int                                 flover_max_concurrency;
  std::mutex                          addingRQ_mtx;

  INIReader                           reader;
  cudaStream_t                        flover_stream;
  IAllocator*                         flover_allocator;
  cublasMMWrapper*                    flover_cublas;
  NcclParam                           flover_tensor_para;
  NcclParam                           flover_pipeline_para;
  std::vector<NcclParam>              sub_tensor_para;
  std::vector<NcclParam>              sub_pipeline_para;
  
  GptJWeight<T>*                      model_weights;
  GptJPreprocessing<T>*               preprocessing;
  GptJInference<T>*                   inference_stream;
  GptJBuffer<T>*                      working_buffer;

  LlamaWeight<T>*                     llama_model_weights;
  LlamaPreprocessing<T>*              llama_preprocessing;
  LlamaInference<T>*                  llama_inference_stream;
  LlamaBuffer<T>*                     llama_working_buffer;

  RequestQueue                        reque;
  InferenceQueue                      inque;
  int                                 task_id;
  int                                 add_req_id = 0;
  std::mutex                          add_req_mtx;

  std::mutex                          RequestMap_mtx;
  std::mutex                          FetchingAndPreprocessing_mtx;

  std::unordered_map<int, std::unique_ptr<std::unordered_map<std::string, Tensor>>> request_input;
  std::unordered_map<int, std::unique_ptr<std::unordered_map<std::string, Tensor>>> request_output;
  tbb::concurrent_queue<int>                                                        request_formatted;

};


}