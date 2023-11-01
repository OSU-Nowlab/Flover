#include "Flover.h"
#include "src/flover/kernels/decoding_kernels.h"
#include "src/flover/kernels/gpt_kernels.h"
#include "src/flover/kernels/bert_preprocess_kernels.h"
#include "src/flover/layers/attention_layers/DecoderSelfAttentionLayer.h"

namespace flover {
template<typename T>
void Flover<T>::test() {
  
  while (true) {
    std::unique_lock<std::mutex> lock(RequestMap_mtx);
    if (!request_input.empty()) {
      for (const auto& pair : request_input) {
        int key = pair.first;
        std::unordered_map<std::string, Tensor>* inner_map_ptr = pair.second.get();

        int my_task_id = inner_map_ptr->at("my_task_id").getVal<int>();
        fprintf(stdout, "test: %d\n", my_task_id);
      }
    }
  }
}


template<typename T>
void Flover<T>::serve(int* stop_flag, int setDeviceNo, std::promise<void> &p_fetching, std::promise<void> &p_inference) {
  log("Flover starts serving...");
  std::thread TR_InferenceStream(&Flover<T>::InferenceStream, this, stop_flag, setDeviceNo, std::ref(p_inference));
  TR_InferenceStream.detach();

  std::thread TR_FetchingAndPreprocessing(&Flover<T>::FetchingAndPreprocessing, this, stop_flag, setDeviceNo, std::ref(p_fetching));
  TR_FetchingAndPreprocessing.detach();
  // TRs_FetchingAndPreprocessing.emplace_back(std::move(TR_FetchingAndPreprocessing));
}

template<typename T>
void Flover<T>::FetchingAndPreprocessing(int* stop_flag, int setDeviceNo, std::promise<void> &p) {
  const std::string model_type = reader.Get("model_specification", "model_type");

  check_cuda_error(cudaSetDevice(setDeviceNo));

  while(*stop_flag == 0) {
    RequestInfo _req;
    
    if (reque.get(_req)) {
      std::thread _TR([=, this](){
        this->new_req = true;
        this->added   = false;
        int device;
        check_cuda_error(cudaSetDevice(setDeviceNo));
        check_cuda_error(cudaGetDevice(&device));
        
        if (model_type == "gptj_6b") {
          const size_t stop_words_len = _req.stop_words.size() / 2;
          
          int total_output_len = _req.total_seq_len;
          std::vector<int>* start_ids = new std::vector<int>(_req.request_batch_size, _req.start_id);
          std::vector<int>* end_ids   = new std::vector<int>(_req.request_batch_size, _req.end_id);

          std::unique_lock<std::mutex> lock(FetchingAndPreprocessing_mtx);
          
          cudaH2Dcpy(working_buffer->coll_d_bad_words->at(_req.unique_task_id), _req.bad_words.data(), _req.bad_words.size());
          cudaH2Dcpy(working_buffer->coll_d_stop_words->at(_req.unique_task_id), _req.tiled_stop_words.data(), _req.tiled_stop_words.size());
          cudaH2Dcpy(working_buffer->coll_d_input_ids->at(_req.unique_task_id), _req.v_start_ids.data(), _req.request_batch_size * _req.max_input_len);
          cudaH2Dcpy(working_buffer->coll_d_input_lengths->at(_req.unique_task_id), _req.v_start_lengths.data(), _req.request_batch_size);

          std::vector<uint32_t>* output_seq_len = new std::vector<uint32_t>(_req.request_batch_size, total_output_len);
          
          unsigned long long* random_seed   = new unsigned long long(_req.random_seed);
          int*   my_id                      = new int(_req.unique_task_id);
          float* temperature                = new float(_req.temperature);
          float* len_penalty                = new float(_req.len_penalty);
          float* min_length                 = new float(_req.min_length);
          float* repetition_penalty         = new float(_req.repetition_penalty);
          float* presence_penalty           = new float(_req.presence_penalty);
          float* beam_search_diversity_rate = new float(_req.beam_search_diversity_rate);
          float* top_p                      = new float(_req.top_p);
          float* top_k                      = new float(_req.top_k);
          float* memory_len                 = new float(_req.memory_len);


          auto input_tensors_map = std::unordered_map<std::string, Tensor>{
              {"input_ids",
              Tensor{MEMORY_GPU, TYPE_INT32, std::vector<size_t>{_req.request_batch_size, (size_t)_req.max_input_len}, working_buffer->coll_d_input_ids->at(_req.unique_task_id)}},
              {"input_lengths", Tensor{MEMORY_GPU, TYPE_INT32, std::vector<size_t>{_req.request_batch_size}, working_buffer->coll_d_input_lengths->at(_req.unique_task_id)}},
              // NOTE: if you need prefix prompts, remember to add prefix_prompt_task_ids here
              // {"prompt_learning_task_name_ids", Tensor{MEMORY_CPU, TYPE_INT32, std::vector<size_t>{request_batch_size},
              // prefix_prompt_task_ids.data()}},
              {"output_seq_len",
              Tensor{MEMORY_CPU, TYPE_UINT32, std::vector<size_t>{_req.request_batch_size}, output_seq_len->data()}},
              {"temperature", Tensor{MEMORY_CPU, TYPE_FP32, std::vector<size_t>{1}, temperature}},
              {"len_penalty", Tensor{MEMORY_CPU, TYPE_FP32, std::vector<size_t>{1}, len_penalty}},
              {"min_length", Tensor{MEMORY_CPU, TYPE_INT32, std::vector<size_t>{1}, min_length}},
              {"start_id", Tensor{MEMORY_CPU, TYPE_INT32, std::vector<size_t>{_req.request_batch_size}, start_ids->data()}},
              {"end_id", Tensor{MEMORY_CPU, TYPE_INT32, std::vector<size_t>{_req.request_batch_size}, end_ids->data()}}};
          if (_req.repetition_penalty != 1.0f) {
              input_tensors_map.insert(
                  {"repetition_penalty", Tensor{MEMORY_CPU, TYPE_FP32, std::vector<size_t>{1}, repetition_penalty}});
          }
          if (_req.presence_penalty != 0.0f) {
              input_tensors_map.insert(
                  {"presence_penalty", Tensor{MEMORY_CPU, TYPE_FP32, std::vector<size_t>{1}, presence_penalty}});
          }

          if (!_req.bad_words.empty()) {
              input_tensors_map.insert(
                  {"bad_words_list", Tensor{MEMORY_GPU, TYPE_INT32, {2, _req.bad_words.size() / 2}, working_buffer->coll_d_bad_words->at(_req.unique_task_id)}});
          }
          if (stop_words_len != 0) {
              input_tensors_map.insert(
                  {"stop_words_list", Tensor{MEMORY_GPU, TYPE_INT32, {_req.request_batch_size, 2, stop_words_len}, working_buffer->coll_d_stop_words->at(_req.unique_task_id)}});
          }
          if (_req.top_k == 0 && _req.top_p == 0.0f) {
              FT_CHECK(_req.beam_width > 1);
              input_tensors_map.insert({"beam_search_diversity_rate",
                                  Tensor{MEMORY_CPU, TYPE_FP32, std::vector<size_t>{1}, beam_search_diversity_rate}});
          }
          else {
              input_tensors_map.insert({"random_seed", Tensor{MEMORY_CPU, TYPE_UINT64, std::vector<size_t>{1}, random_seed}});
              if (_req.top_p != 0.0f) {
                  input_tensors_map.insert({"runtime_top_p", Tensor{MEMORY_CPU, TYPE_FP32, std::vector<size_t>{1}, top_p}});
              }
              if (_req.top_k != 0) {
                  input_tensors_map.insert({"runtime_top_k", Tensor{MEMORY_CPU, TYPE_UINT32, std::vector<size_t>{1}, top_k}});
              }
          }
          if (_req.memory_len > 0) {
              input_tensors_map.insert({"memory_len", {MEMORY_CPU, TYPE_UINT32, {1}, memory_len}});
          }

          input_tensors_map.insert({"my_task_id", {MEMORY_CPU, TYPE_INT32, {1}, my_id}});
          auto input_tensors = std::make_unique<std::unordered_map<std::string, Tensor>>(std::move(input_tensors_map));

          auto output_tensors_map = std::unordered_map<std::string, Tensor>{
          {"output_ids",
          Tensor{MEMORY_GPU,
                  TYPE_INT32,
                  std::vector<size_t>{_req.request_batch_size, _req.beam_width, (size_t)total_output_len},
                  working_buffer->coll_d_output_ids->at(_req.unique_task_id)}},
          {"sequence_length",
          Tensor{MEMORY_GPU, TYPE_INT32, std::vector<size_t>{_req.request_batch_size, _req.beam_width}, working_buffer->coll_d_sequence_lengths->at(_req.unique_task_id)}},
          {"output_log_probs",
          Tensor{MEMORY_GPU,
                  TYPE_FP32,
                  std::vector<size_t>{(size_t)_req.generation_len, _req.request_batch_size, _req.beam_width},
                  nullptr}}};
          auto output_tensors = std::make_unique<std::unordered_map<std::string, Tensor>>(std::move(output_tensors_map));
          

          /* Assign pre initialized NCCL communicator */
          NcclParam _tensor_para(sub_tensor_para[*my_id]);
          NcclParam _pipeline_para(sub_pipeline_para[*my_id]);

          cudaStream_t _stream;
          cudaStreamCreate(&_stream);
          cublasHandle_t   _cublas_handle;
          cublasLtHandle_t _cublaslt_handle;
          cublasCreate(&_cublas_handle);
          cublasLtCreate(&_cublaslt_handle);
          cublasSetStream(_cublas_handle, _stream);
          cublasAlgoMap* _cublas_algo_map      = new cublasAlgoMap("gemm_config.in");

          std::mutex*    _cublas_wrapper_mutex = new std::mutex();
          cublasMMWrapper _cublas_wrapper      = cublasMMWrapper(_cublas_handle, _cublaslt_handle, _stream, _cublas_algo_map, _cublas_wrapper_mutex, nullptr);
          if (std::is_same<T, half>::value) {
              _cublas_wrapper.setGemmConfig(CUDA_R_16F, CUDA_R_16F, CUDA_R_16F, CUDA_R_32F);
          }
        #ifdef ENABLE_BF16
          else if (std::is_same<T, __nv_bfloat16>::value) {
              _cublas_wrapper.setBF16GemmConfig();
          }
        #endif
          else if (std::is_same<T, float>::value) {
              _cublas_wrapper.setFP32GemmConfig();
          }
          // Refer to https://docs.nvidia.com/cuda/cublas/#cublassetworkspace
          // We do not set this workspace as on non-Hopper GPU, as the pre setting value in cuda_utils.h is 32MiB which might cause Error 700 (Illegal Memory Access)
          // _cublas_wrapper.cublas_workspace_    = working_buffer->cublas_workspace + CUBLAS_WORKSPACE_SIZE * (1 + *my_id);

          std::unique_lock<std::mutex> RequestMap_lock(RequestMap_mtx);
          request_input.insert({*my_id, std::move(input_tensors)});
          request_output.insert({*my_id, std::move(output_tensors)});
          request_formatted.push(*my_id);
          std::unordered_map<std::string, Tensor>* _output_tensors = request_output.at(*my_id).get();
          std::unordered_map<std::string, Tensor>* _input_tensors = request_input.at(*my_id).get();

          RequestMap_lock.unlock();
          // lock.unlock();
          // log("Rank ", flover_tensor_para.rank_, " id ", *my_id, " start running");
          func_preprocessing(*my_id, _stream, _tensor_para, _pipeline_para, &_cublas_wrapper, _output_tensors, _input_tensors);
          
          // log("Rank ", flover_tensor_para.rank_, " id ", *my_id, " waiting adding");
          inque.add<T>(_req, _output_tensors, _input_tensors,
                      working_buffer->coll_d_bad_words->at(_req.unique_task_id),
                      working_buffer->coll_d_stop_words->at(_req.unique_task_id),
                      working_buffer->coll_d_input_ids->at(_req.unique_task_id),
                      working_buffer->coll_d_input_lengths->at(_req.unique_task_id),
                      working_buffer->coll_d_output_ids->at(_req.unique_task_id),
                      working_buffer->coll_d_sequence_lengths->at(_req.unique_task_id));
          // log("Rank ", flover_tensor_para.rank_, " id ", *my_id, " added");
          ftNcclStreamSynchronize(_tensor_para, _pipeline_para, _stream);
        }
        else if (model_type.substr(0, 5) == "llama") {
          const size_t stop_words_len = _req.stop_words.size() / 2;
          
          int total_output_len = _req.total_seq_len;
          std::vector<int>* start_ids = new std::vector<int>(_req.request_batch_size, _req.start_id);
          std::vector<int>* end_ids   = new std::vector<int>(_req.request_batch_size, _req.end_id);
          
          cudaH2Dcpy(llama_working_buffer->coll_d_bad_words->at(_req.unique_task_id), _req.bad_words.data(), _req.bad_words.size());
          cudaH2Dcpy(llama_working_buffer->coll_d_stop_words->at(_req.unique_task_id), _req.tiled_stop_words.data(), _req.tiled_stop_words.size());
          cudaH2Dcpy(llama_working_buffer->coll_d_input_ids->at(_req.unique_task_id), _req.v_start_ids.data(), _req.request_batch_size * _req.max_input_len);
          cudaH2Dcpy(llama_working_buffer->coll_d_input_lengths->at(_req.unique_task_id), _req.v_start_lengths.data(), _req.request_batch_size);

          std::vector<uint32_t>* output_seq_len = new std::vector<uint32_t>(_req.request_batch_size, total_output_len);
          
          unsigned long long* random_seed   = new unsigned long long(_req.random_seed);
          int*   my_id                      = new int(_req.unique_task_id);
          float* temperature                = new float(_req.temperature);
          float* len_penalty                = new float(_req.len_penalty);
          float* min_length                 = new float(_req.min_length);
          float* repetition_penalty         = new float(_req.repetition_penalty);
          float* presence_penalty           = new float(_req.presence_penalty);
          float* beam_search_diversity_rate = new float(_req.beam_search_diversity_rate);
          float* top_p                      = new float(_req.top_p);
          int* top_k                        = new int(_req.top_k);
          float* memory_len                 = new float(_req.memory_len);


          auto input_tensors_map = std::unordered_map<std::string, Tensor>{
              {"input_ids",
              Tensor{MEMORY_GPU, TYPE_INT32, std::vector<size_t>{_req.request_batch_size, (size_t)_req.max_input_len}, llama_working_buffer->coll_d_input_ids->at(_req.unique_task_id)}},
              {"input_lengths", Tensor{MEMORY_GPU, TYPE_INT32, std::vector<size_t>{_req.request_batch_size}, llama_working_buffer->coll_d_input_lengths->at(_req.unique_task_id)}},
              // NOTE: if you need prefix prompts, remember to add prefix_prompt_task_ids here
              // {"prompt_learning_task_name_ids", Tensor{MEMORY_CPU, TYPE_INT32, std::vector<size_t>{request_batch_size},
              // prefix_prompt_task_ids.data()}},
              {"output_seq_len",
              Tensor{MEMORY_CPU, TYPE_UINT32, std::vector<size_t>{_req.request_batch_size}, output_seq_len->data()}},
              {"temperature", Tensor{MEMORY_CPU, TYPE_FP32, std::vector<size_t>{1}, temperature}},
              {"len_penalty", Tensor{MEMORY_CPU, TYPE_FP32, std::vector<size_t>{1}, len_penalty}},
              {"min_length", Tensor{MEMORY_CPU, TYPE_INT32, std::vector<size_t>{1}, min_length}},
              {"start_id", Tensor{MEMORY_CPU, TYPE_INT32, std::vector<size_t>{_req.request_batch_size}, start_ids->data()}},
              {"end_id", Tensor{MEMORY_CPU, TYPE_INT32, std::vector<size_t>{_req.request_batch_size}, end_ids->data()}}};
          if (_req.repetition_penalty != 1.0f) {
              input_tensors_map.insert(
                  {"repetition_penalty", Tensor{MEMORY_CPU, TYPE_FP32, std::vector<size_t>{1}, repetition_penalty}});
          }
          if (_req.presence_penalty != 0.0f) {
              input_tensors_map.insert(
                  {"presence_penalty", Tensor{MEMORY_CPU, TYPE_FP32, std::vector<size_t>{1}, presence_penalty}});
          }

          if (!_req.bad_words.empty()) {
              input_tensors_map.insert(
                  {"bad_words_list", Tensor{MEMORY_GPU, TYPE_INT32, {2, _req.bad_words.size() / 2}, llama_working_buffer->coll_d_bad_words->at(_req.unique_task_id)}});
          }
          if (stop_words_len != 0) {
              input_tensors_map.insert(
                  {"stop_words_list", Tensor{MEMORY_GPU, TYPE_INT32, {_req.request_batch_size, 2, stop_words_len}, llama_working_buffer->coll_d_stop_words->at(_req.unique_task_id)}});
          }
          if (_req.top_k == 0 && _req.top_p == 0.0f) {
              FT_CHECK(_req.beam_width > 1);
              input_tensors_map.insert({"beam_search_diversity_rate",
                                  Tensor{MEMORY_CPU, TYPE_FP32, std::vector<size_t>{1}, beam_search_diversity_rate}});
          }
          else {
              input_tensors_map.insert({"random_seed", Tensor{MEMORY_CPU, TYPE_UINT64, std::vector<size_t>{1}, random_seed}});
              if (_req.top_p != 0.0f) {
                  input_tensors_map.insert({"runtime_top_p", Tensor{MEMORY_CPU, TYPE_FP32, std::vector<size_t>{1}, top_p}});
              }
              if (_req.top_k != 0) {
                  input_tensors_map.insert({"runtime_top_k", Tensor{MEMORY_CPU, TYPE_UINT32, std::vector<size_t>{1}, top_k}});
              }
          }
          if (_req.memory_len > 0) {
              input_tensors_map.insert({"memory_len", {MEMORY_CPU, TYPE_UINT32, {1}, memory_len}});
          }

          input_tensors_map.insert({"my_task_id", {MEMORY_CPU, TYPE_INT32, {1}, my_id}});
          auto input_tensors = std::make_unique<std::unordered_map<std::string, Tensor>>(std::move(input_tensors_map));

          auto output_tensors_map = std::unordered_map<std::string, Tensor>{
          {"output_ids",
          Tensor{MEMORY_GPU,
                  TYPE_INT32,
                  std::vector<size_t>{_req.request_batch_size, _req.beam_width, (size_t)total_output_len},
                  llama_working_buffer->coll_d_output_ids->at(_req.unique_task_id)}},
          {"sequence_length",
          Tensor{MEMORY_GPU, TYPE_INT32, std::vector<size_t>{_req.request_batch_size, _req.beam_width}, llama_working_buffer->coll_d_sequence_lengths->at(_req.unique_task_id)}},
          {"output_log_probs",
          Tensor{MEMORY_GPU,
                  TYPE_FP32,
                  std::vector<size_t>{(size_t)_req.generation_len, _req.request_batch_size, _req.beam_width},
                  nullptr}}};
          auto output_tensors = std::make_unique<std::unordered_map<std::string, Tensor>>(std::move(output_tensors_map));
          

          /* Assign pre initialized NCCL communicator */
          NcclParam _tensor_para(sub_tensor_para[*my_id]);
          NcclParam _pipeline_para(sub_pipeline_para[*my_id]);

          cudaStream_t _stream;
          cudaStreamCreate(&_stream);
          cublasHandle_t   _cublas_handle;
          cublasLtHandle_t _cublaslt_handle;
          cublasCreate(&_cublas_handle);
          cublasLtCreate(&_cublaslt_handle);
          cublasSetStream(_cublas_handle, _stream);
          cublasAlgoMap* _cublas_algo_map      = new cublasAlgoMap("gemm_config.in");

          std::mutex*    _cublas_wrapper_mutex = new std::mutex();
          cublasMMWrapper _cublas_wrapper      = cublasMMWrapper(_cublas_handle, _cublaslt_handle, _stream, _cublas_algo_map, _cublas_wrapper_mutex, nullptr);
          if (std::is_same<T, half>::value) {
              _cublas_wrapper.setGemmConfig(CUDA_R_16F, CUDA_R_16F, CUDA_R_16F, CUDA_R_32F);
          }
        #ifdef ENABLE_BF16
          else if (std::is_same<T, __nv_bfloat16>::value) {
              _cublas_wrapper.setBF16GemmConfig();
          }
        #endif
          else if (std::is_same<T, float>::value) {
              _cublas_wrapper.setFP32GemmConfig();
          }
          // Refer to https://docs.nvidia.com/cuda/cublas/#cublassetworkspace
          // We do not set this workspace as on non-Hopper GPU, as the pre setting value in cuda_utils.h is 32MiB which might cause Error 700 (Illegal Memory Access)
          // _cublas_wrapper.cublas_workspace_    = working_buffer->cublas_workspace + CUBLAS_WORKSPACE_SIZE * (1 + *my_id);

          std::unique_lock<std::mutex> RequestMap_lock(RequestMap_mtx);
          request_input.insert({*my_id, std::move(input_tensors)});
          request_output.insert({*my_id, std::move(output_tensors)});
          request_formatted.push(*my_id);
          std::unordered_map<std::string, Tensor>* _output_tensors = request_output.at(*my_id).get();
          std::unordered_map<std::string, Tensor>* _input_tensors = request_input.at(*my_id).get();

          RequestMap_lock.unlock();
          
          // log("Rank ", flover_tensor_para.rank_, " id ", *my_id, " start running");
          // func_preprocessing(*my_id, _stream, _tensor_para, _pipeline_para, &_cublas_wrapper, _output_tensors, _input_tensors);
          
          // log("Rank ", flover_tensor_para.rank_, " id ", *my_id, " waiting adding");
          inque.add<T>(_req, _output_tensors, _input_tensors,
                      llama_working_buffer->coll_d_bad_words->at(_req.unique_task_id),
                      llama_working_buffer->coll_d_stop_words->at(_req.unique_task_id),
                      llama_working_buffer->coll_d_input_ids->at(_req.unique_task_id),
                      llama_working_buffer->coll_d_input_lengths->at(_req.unique_task_id),
                      llama_working_buffer->coll_d_output_ids->at(_req.unique_task_id),
                      llama_working_buffer->coll_d_sequence_lengths->at(_req.unique_task_id));
          // log("Rank ", flover_tensor_para.rank_, " id ", *my_id, " added");
          this->added   = true;
        }
      });
      TRs_FetchingAndPreprocessing.emplace_back(std::move(_TR));
    }
  }
  for (auto& tr : TRs_FetchingAndPreprocessing){
    tr.join();
  }
  p.set_value();
}


template<typename T>
void Flover<T>::func_preprocessing(int my_id, 
                                   cudaStream_t stream, 
                                   NcclParam tensor_para, 
                                   NcclParam pipeline_para, 
                                   cublasMMWrapper* cublas_wrapper,
                                   std::unordered_map<std::string, Tensor>*       output_tensors,
                                   const std::unordered_map<std::string, Tensor>* input_tensors) {
  const std::string model_type = reader.Get("model_specification", "model_type");
  if (model_type == "gptj_6b") {
    preprocessing = gptj_preprocessing<T>(reader, stream, tensor_para, pipeline_para, cublas_wrapper);
    // std::cout << "Thread ID: " << std::this_thread::get_id() << "\n";
    // std::this_thread::sleep_for(std::chrono::seconds(5));
    // preprocessing->decoder_normed_input_   = working_buffer + my_id * ;
    const std::string model_name           = reader.Get("model_specification", "model_type");
    const size_t      max_concurrency      = reader.GetInteger("model_specification", "max_concurrency");
    const size_t      per_batch_size       = reader.GetInteger("model_specification", "per_batch_size");
    const size_t      total_seq_len          = reader.GetInteger("model_specification", "max_seq_len");
    const size_t      fixed_input_len      = reader.GetInteger("model_specification", "fixed_input_len");
    const size_t      fixed_prompt_len     = reader.GetInteger("model_specification", "fixed_prompt_len");
    const size_t      beam_width           = reader.GetInteger("model_specification", "beam_width");
    int               tensor_para_size     = reader.GetInteger("model_specification", "tensor_para_size");
    int               pipeline_para_size   = reader.GetInteger("model_specification", "pipeline_para_size");
    const size_t      max_batch_size       = per_batch_size;

    const size_t      head_num             = reader.GetInteger(model_name, "head_num");
    const size_t      size_per_head        = reader.GetInteger(model_name, "size_per_head");
    const size_t      vocab_size           = reader.GetInteger(model_name, "vocab_size");
    const size_t      decoder_layers       = reader.GetInteger(model_name, "decoder_layers");
    const size_t      rotary_embedding_dim = reader.GetInteger(model_name, "rotary_embedding");
    const size_t      inter_size           = reader.GetInteger(model_name, "inter_size");
    const size_t      hidden_units         = head_num * size_per_head;
    
    const size_t      local_head_num       = head_num / tensor_para_size;
    const size_t      local_hidden_units   = head_num * size_per_head / tensor_para_size;

    const int         working_batch_size   = max_batch_size * beam_width;
    const int         context_working_len  = fixed_input_len + fixed_prompt_len;

    const int         start_id             = reader.GetInteger(model_name, "start_id");
    const int         end_id               = reader.GetInteger(model_name, "end_id");
    preprocessing->start_id_               = start_id;
    preprocessing->end_id_                 = end_id;

    preprocessing->decoder_normed_input_   = working_buffer->context_decoder_decoder_normed_input + my_id * working_batch_size * context_working_len * hidden_units;
    preprocessing->self_attn_output_       = working_buffer->context_decoder_self_attn_output + my_id * working_batch_size * context_working_len * hidden_units;
    preprocessing->ffn_output_             = working_buffer->context_decoder_ffn_output + my_id * working_batch_size * context_working_len * hidden_units;
    preprocessing->decoder_layer_output_   = working_buffer->context_decoder_decoder_layer_output + my_id * working_batch_size * context_working_len * hidden_units;
    preprocessing->h_pinned_token_num_ptr_ = working_buffer->context_decoder_h_pinned_token_num_ptr + my_id;
    preprocessing->padding_offset_         = working_buffer->context_decoder_padding_offset + my_id * working_batch_size * context_working_len;
    preprocessing->cu_seqlens_             = working_buffer->context_decoder_cu_seqlens + my_id * (working_batch_size + 1);

    const AttentionType attention_type = getAttentionType<T>(size_per_head,
                                            getSMVersion(),
                                            true,   // remove_padding
                                            0,      // gpt supports any-seq-length fmha
                                            true,   // is_fuse
                                            false,  // with_relative_position_bias
                                            true);  // causal_mask
    
    const auto        type_size                 = sizeof(T);
    int               int8_mode                 = 0;
    bool              attn_flag                 = attention_type != AttentionType::FUSED_MHA;
    const bool        is_context_qk_buf_float   = (std::getenv("CONTEXT_ATTENTION_BMM1_HALF_ACCUM") == nullptr || std::string(std::getenv("CONTEXT_ATTENTION_BMM1_HALF_ACCUM")) != "ON");

    GptContextAttentionLayer<T>* _temp_         = dynamic_cast<GptContextAttentionLayer<T>*>(preprocessing->self_attention_layer_);
    _temp_->qkv_buf_                            = working_buffer->context_attention_qkv_buf + my_id * 3 * working_batch_size * context_working_len * local_hidden_units;
    _temp_->q_buf_2_                            = working_buffer->context_attention_q_buf_2 + my_id * working_batch_size * context_working_len * local_hidden_units;
    _temp_->k_buf_2_                            = working_buffer->context_attention_k_buf_2 + my_id * working_batch_size * context_working_len * local_hidden_units;
    _temp_->v_buf_2_                            = working_buffer->context_attention_v_buf_2 + my_id * working_batch_size * context_working_len * local_hidden_units;
    if (attn_flag) {
        _temp_->qk_buf_                         = working_buffer->context_attention_qk_buf + my_id * working_batch_size * context_working_len * context_working_len * local_head_num;
    }
    _temp_->qkv_buf_2_                          = working_buffer->context_attention_qkv_buf_2 + my_id * working_batch_size * context_working_len * local_hidden_units;
    _temp_->qkv_buf_3_                          = working_buffer->context_attention_qkv_buf_3 + my_id * working_batch_size * context_working_len * local_hidden_units;

    if (is_context_qk_buf_float && attn_flag) {
      _temp_->qk_buf_float_                     = working_buffer->context_attention_qk_buf_float + my_id * working_batch_size * context_working_len * context_working_len * local_head_num;
    }

    // used by each ctxt
    std::shared_ptr<CutlassFpAIntBGemmRunner<T, uint8_t>> template_weight_only_int8_fc_runner(int8_mode == 1 ? std::make_shared<CutlassFpAIntBGemmRunner<T, uint8_t>>() : nullptr);
    std::shared_ptr<CutlassInt8GemmRunner<T>>             template_int8_fc_runner(int8_mode == 2 ? std::make_shared<CutlassInt8GemmRunner<T>>() : nullptr);
    if (int8_mode == 1){
        const int max_size                                     = std::max(hidden_units, 3 * local_hidden_units);
        size_t    context_attention_mixed_gemm_ws_bytes        = template_weight_only_int8_fc_runner->getWorkspaceSize(working_batch_size * context_working_len, max_size, max_size);
        _temp_->mixed_gemm_workspace_                          = working_buffer->context_attention_mixed_gemm_workspace + my_id * context_attention_mixed_gemm_ws_bytes;
    }
    else if (int8_mode == 2) {
        const int max_size                                     = std::max(hidden_units, 3 * local_hidden_units);
        size_t    context_attention_int8_gemm_ws_bytes         = template_int8_fc_runner->getWorkspaceSize(working_batch_size * context_working_len, max_size, max_size);
        _temp_->int8_gemm_workspace_                           = working_buffer->context_attention_int8_gemm_workspace + my_id * working_buffer->context_attention_int8_gemm_ws_bytes;
    }

    
    const auto type_size_1 = int8_mode == 2 ? sizeof(int8_t) : sizeof(T);
    const auto token_num   = working_batch_size * context_working_len;
    
    preprocessing->ffn_layer_->inter_buf_                  = working_buffer->context_ffn_inter_buf + my_id * token_num * inter_size;
    if (int8_mode == 1) {
        const int max_size                                 = std::max(hidden_units, inter_size);
        size_t    context_ffn_mixed_gemm_ws_bytes          = template_weight_only_int8_fc_runner->getWorkspaceSize(token_num, max_size, max_size);
        preprocessing->ffn_layer_->mixed_gemm_workspace_   = working_buffer->context_ffn_mixed_gemm_workspace + my_id * working_buffer->context_ffn_mixed_gemm_ws_bytes;
    }
    else if (int8_mode == 2) {
        const int max_size                                 = std::max(hidden_units, inter_size);
        size_t    context_ffn_int8_gemm_ws_bytes           = template_int8_fc_runner->getWorkspaceSize(token_num, max_size, max_size);
        preprocessing->ffn_layer_->int8_gemm_workspace_    = working_buffer->context_ffn_int8_gemm_workspace + my_id * working_buffer->context_ffn_int8_gemm_ws_bytes;
    }
    

    // shared
    const size_t max_cache_seq_len             = total_seq_len;
    const size_t max_prefix_soft_prompt_length = fixed_prompt_len;
    size_t       max_input_len                 = fixed_input_len + max_prefix_soft_prompt_length;

    const size_t batchxbeam                    = working_batch_size;
    const size_t self_cache_size               = (decoder_layers / pipeline_para_size) * batchxbeam * max_cache_seq_len * local_hidden_units;

    int          local_vacab_size              = std::ceil(vocab_size / 1.f / tensor_para_size);
    if (std::is_same<half, T>::value) {
                 local_vacab_size              = std::ceil(local_vacab_size / 8.f) * 8;
    }
    size_t       vocab_size_padded             = (size_t)local_vacab_size * tensor_para_size;

    if (vocab_size != vocab_size_padded) {
        preprocessing->padded_embedding_kernel_               = working_buffer->padded_embedding_kernel_;
        preprocessing->padded_embedding_kernel_ptr_           = preprocessing->padded_embedding_kernel_;
        preprocessing->padded_embedding_bias_                 = working_buffer->padded_embedding_bias_;
        preprocessing->padded_embedding_bias_ptr_             = preprocessing->padded_embedding_bias_;
    }
    
    preprocessing->input_attention_mask_                      = working_buffer->input_attention_mask_ + my_id * batchxbeam * total_seq_len * max_cache_seq_len;
    preprocessing->decoder_input_buf_                         = working_buffer->decoder_input_buf_ + my_id * batchxbeam * hidden_units;
    preprocessing->decoder_output_buf_                        = working_buffer->decoder_output_buf_ + my_id * batchxbeam * hidden_units;
    preprocessing->normed_decoder_output_buf_                 = working_buffer->normed_decoder_output_buf_ + my_id * batchxbeam * hidden_units;

    preprocessing->logits_buf_                                = working_buffer->logits_buf_ + my_id * batchxbeam * vocab_size_padded;
    preprocessing->nccl_logits_buf_                           = working_buffer->nccl_logits_buf_ + my_id * batchxbeam * vocab_size_padded;
    preprocessing->cum_log_probs_                             = working_buffer->cum_log_probs_ + my_id * batchxbeam;
    preprocessing->sequence_lengths_                          = working_buffer->sequence_lengths_ + my_id * batchxbeam;

    preprocessing->finished_buf_                              = working_buffer->finished_buf_ + my_id * batchxbeam;
    preprocessing->h_finished_buf_                            = working_buffer->h_finished_buf_ + my_id * batchxbeam;

    preprocessing->key_cache_                                 = working_buffer->key_cache_ + my_id * self_cache_size;
    preprocessing->value_cache_                               = working_buffer->value_cache_ + my_id * self_cache_size;

    if (beam_width > 1) {
        preprocessing->cache_indirections_[0]                 = working_buffer->cache_indirections_[0] + my_id * batchxbeam * max_cache_seq_len;
        preprocessing->cache_indirections_[1]                 = working_buffer->cache_indirections_[1] + my_id * batchxbeam * max_cache_seq_len;
    }

    preprocessing->tiled_total_padding_count_                 = working_buffer->tiled_total_padding_count_ + my_id * batchxbeam;

    preprocessing->prompt_learning_weight_batch_              = working_buffer->prompt_learning_weight_batch_ + my_id * batchxbeam;
    preprocessing->tiled_prompt_lengths_buf_                  = working_buffer->tiled_prompt_lengths_buf_ + my_id * batchxbeam;

    preprocessing->tiled_input_ids_buf_                       = working_buffer->tiled_input_ids_buf_ + my_id * batchxbeam * max_input_len;
    preprocessing->tiled_input_lengths_buf_                   = working_buffer->tiled_input_lengths_buf_ + my_id * batchxbeam;
    preprocessing->transposed_output_ids_buf_                 = working_buffer->transposed_output_ids_buf_ + my_id * batchxbeam * total_seq_len;
    preprocessing->output_ids_buf_                            = working_buffer->output_ids_buf_ + my_id * batchxbeam * total_seq_len;
    preprocessing->parent_ids_buf_                            = working_buffer->parent_ids_buf_ + my_id * batchxbeam * total_seq_len;

    preprocessing->start_ids_buf_                             = working_buffer->start_ids_buf_ + my_id * max_batch_size;
    preprocessing->end_ids_buf_                               = working_buffer->end_ids_buf_ + my_id * max_batch_size;
    preprocessing->seq_limit_len_                             = working_buffer->seq_limit_len_ + my_id * max_batch_size;
    
    preprocessing->context_decoder_input_buf_                 = working_buffer->context_decoder_input_buf_ + my_id * batchxbeam * max_input_len * hidden_units;
    preprocessing->context_decoder_output_buf_                = working_buffer->context_decoder_output_buf_ + my_id * batchxbeam * max_input_len * hidden_units;
    preprocessing->output_log_probs_buf_                      = working_buffer->output_log_probs_buf_ + my_id * batchxbeam * total_seq_len;

    preprocessing->generation_should_stop_                    = working_buffer->generation_should_stop_;
    fprintf(stdout, "%d , size asgn %d\n", my_id, batchxbeam * max_cache_seq_len);
    preprocessing->masked_tokens_                             = working_buffer->masked_tokens_ + my_id * batchxbeam * max_cache_seq_len;

    // GptJWeight<T>* _temp_weights           = dynamic_cast<GptJWeight<T>*>(model_weights);
    preprocessing->run(output_tensors, input_tensors, model_weights);
  }
  else if (model_type.substr(0, 5) == "llama") {
    llama_preprocessing = llama_init_preprocessing<T>(reader, stream, tensor_para, pipeline_para, cublas_wrapper);
    // std::cout << "Thread ID: " << std::this_thread::get_id() << "\n";
    // std::this_thread::sleep_for(std::chrono::seconds(5));
    // preprocessing->decoder_normed_input_   = working_buffer + my_id * ;
    const std::string model_name           = reader.Get("model_specification", "model_type");
    const size_t      max_concurrency      = reader.GetInteger("model_specification", "max_concurrency");
    const size_t      per_batch_size       = reader.GetInteger("model_specification", "per_batch_size");
    const size_t      total_seq_len          = reader.GetInteger("model_specification", "max_seq_len");
    const size_t      fixed_input_len      = reader.GetInteger("model_specification", "fixed_input_len");
    const size_t      fixed_prompt_len     = reader.GetInteger("model_specification", "fixed_prompt_len");
    const size_t      beam_width           = reader.GetInteger("model_specification", "beam_width");
    int               tensor_para_size     = reader.GetInteger("model_specification", "tensor_para_size");
    int               pipeline_para_size   = reader.GetInteger("model_specification", "pipeline_para_size");
    const size_t      max_batch_size       = per_batch_size;

    const size_t      head_num             = reader.GetInteger(model_name, "head_num");
    const size_t      size_per_head        = reader.GetInteger(model_name, "size_per_head");
    const size_t      vocab_size           = reader.GetInteger(model_name, "vocab_size");
    const size_t      decoder_layers       = reader.GetInteger(model_name, "decoder_layers");
    const size_t      rotary_embedding_dim = reader.GetInteger(model_name, "rotary_embedding");
    const size_t      inter_size           = reader.GetInteger(model_name, "inter_size");
    const size_t      hidden_units         = head_num * size_per_head;
    const float       shared_contexts_ratio = reader.GetFloat(model_name, "shared_contexts_ratio");
    bool              use_shared_contexts   = (shared_contexts_ratio > 0.0f) && (fixed_input_len >= 1) && (per_batch_size > 1);
    
    const size_t      local_head_num       = head_num / tensor_para_size;
    const size_t      local_hidden_units   = head_num * size_per_head / tensor_para_size;

    const int         working_batch_size   = max_batch_size * beam_width;
    const int         context_working_len  = fixed_input_len + fixed_prompt_len;

    const int         start_id             = reader.GetInteger(model_name, "start_id");
    const int         end_id               = reader.GetInteger(model_name, "end_id");
    llama_preprocessing->start_id_               = start_id;
    llama_preprocessing->end_id_                 = end_id;

    llama_preprocessing->decoder_normed_input_   = llama_working_buffer->context_decoder_decoder_normed_input + my_id * working_batch_size * context_working_len * hidden_units;
    llama_preprocessing->self_attn_output_       = llama_working_buffer->context_decoder_self_attn_output + my_id * working_batch_size * context_working_len * hidden_units;
    llama_preprocessing->ffn_output_             = llama_working_buffer->context_decoder_ffn_output + my_id * working_batch_size * context_working_len * hidden_units;
    llama_preprocessing->decoder_layer_output_   = llama_working_buffer->context_decoder_decoder_layer_output + my_id * working_batch_size * context_working_len * hidden_units;
    llama_preprocessing->h_pinned_token_num_ptr_ = llama_working_buffer->context_decoder_h_pinned_token_num_ptr + my_id;
    llama_preprocessing->padding_offset_         = llama_working_buffer->context_decoder_padding_offset + my_id * working_batch_size * context_working_len;
    llama_preprocessing->cu_seqlens_             = llama_working_buffer->context_decoder_cu_seqlens + my_id * (working_batch_size + 1);
    if (use_shared_contexts) {
      llama_preprocessing->compact_decoder_features_    = llama_working_buffer->context_decoder_compact_decoder_features + my_id * working_batch_size * context_working_len * hidden_units;
      llama_preprocessing->compact_attention_mask_      = llama_working_buffer->context_decoder_compact_attention_mask + my_id * working_batch_size * context_working_len * context_working_len;
      llama_preprocessing->compact_input_lengths_       = llama_working_buffer->context_decoder_compact_input_lengths + my_id * working_batch_size;
      llama_preprocessing->k_cache_layer_               = llama_working_buffer->context_decoder_k_cache_layer + my_id * working_batch_size * context_working_len * hidden_units;
      llama_preprocessing->v_cache_layer_               = llama_working_buffer->context_decoder_v_cache_layer + my_id * working_batch_size * context_working_len * hidden_units;
    }

    const AttentionType attention_type = getAttentionType<T>(size_per_head,
                                            getSMVersion(),
                                            true,   // remove_padding
                                            0,      // gpt supports any-seq-length fmha
                                            true,   // is_fuse
                                            false,  // with_relative_position_bias
                                            true);  // causal_mask
    
    const auto        type_size                 = sizeof(T);
    int               int8_mode                 = 0;
    bool              attn_flag                 = attention_type != AttentionType::FUSED_MHA;
    const bool        is_context_qk_buf_float   = (std::getenv("CONTEXT_ATTENTION_BMM1_HALF_ACCUM") == nullptr || std::string(std::getenv("CONTEXT_ATTENTION_BMM1_HALF_ACCUM")) != "ON");

    GptContextAttentionLayer<T>* _temp_         = dynamic_cast<GptContextAttentionLayer<T>*>(llama_preprocessing->self_attention_layer_);
    _temp_->qkv_buf_                            = llama_working_buffer->context_attention_qkv_buf + my_id * 3 * working_batch_size * context_working_len * local_hidden_units;
    _temp_->q_buf_2_                            = llama_working_buffer->context_attention_q_buf_2 + my_id * working_batch_size * context_working_len * local_hidden_units;
    _temp_->k_buf_2_                            = llama_working_buffer->context_attention_k_buf_2 + my_id * working_batch_size * context_working_len * local_hidden_units;
    _temp_->v_buf_2_                            = llama_working_buffer->context_attention_v_buf_2 + my_id * working_batch_size * context_working_len * local_hidden_units;
    if (attn_flag) {
        _temp_->qk_buf_                         = llama_working_buffer->context_attention_qk_buf + my_id * working_batch_size * context_working_len * context_working_len * local_head_num;
    }
    _temp_->qkv_buf_2_                          = llama_working_buffer->context_attention_qkv_buf_2 + my_id * working_batch_size * context_working_len * local_hidden_units;
    _temp_->qkv_buf_3_                          = llama_working_buffer->context_attention_qkv_buf_3 + my_id * working_batch_size * context_working_len * local_hidden_units;

    if (is_context_qk_buf_float && attn_flag) {
      _temp_->qk_buf_float_                     = llama_working_buffer->context_attention_qk_buf_float + my_id * working_batch_size * context_working_len * context_working_len * local_head_num;
    }

    // used by each ctxt
    std::shared_ptr<CutlassFpAIntBGemmRunner<T, uint8_t>> template_weight_only_int8_fc_runner(int8_mode == 1 ? std::make_shared<CutlassFpAIntBGemmRunner<T, uint8_t>>() : nullptr);
    std::shared_ptr<CutlassInt8GemmRunner<T>>             template_int8_fc_runner(int8_mode == 2 ? std::make_shared<CutlassInt8GemmRunner<T>>() : nullptr);
    if (int8_mode == 1){
        const int max_size                                     = std::max(hidden_units, 3 * local_hidden_units);
        size_t    context_attention_mixed_gemm_ws_bytes        = template_weight_only_int8_fc_runner->getWorkspaceSize(working_batch_size * context_working_len, max_size, max_size);
        _temp_->mixed_gemm_workspace_                          = llama_working_buffer->context_attention_mixed_gemm_workspace + my_id * context_attention_mixed_gemm_ws_bytes;
    }
    else if (int8_mode == 2) {
        const int max_size                                     = std::max(hidden_units, 3 * local_hidden_units);
        size_t    context_attention_int8_gemm_ws_bytes         = template_int8_fc_runner->getWorkspaceSize(working_batch_size * context_working_len, max_size, max_size);
        _temp_->int8_gemm_workspace_                           = llama_working_buffer->context_attention_int8_gemm_workspace + my_id * llama_working_buffer->context_attention_int8_gemm_ws_bytes;
    }

    
    const auto type_size_1 = int8_mode == 2 ? sizeof(int8_t) : sizeof(T);
    const auto token_num   = working_batch_size * context_working_len;
    
    llama_preprocessing->ffn_layer_->inter_buf_                  = llama_working_buffer->context_ffn_inter_buf + my_id * token_num * inter_size;
    llama_preprocessing->ffn_layer_->inter_buf_2_                = llama_working_buffer->context_ffn_inter_buf_2 + my_id * token_num * inter_size;
    if (int8_mode == 1) {
        const int max_size                                 = std::max(hidden_units, inter_size);
        size_t    context_ffn_mixed_gemm_ws_bytes          = template_weight_only_int8_fc_runner->getWorkspaceSize(token_num, max_size, max_size);
        llama_preprocessing->ffn_layer_->mixed_gemm_workspace_   = llama_working_buffer->context_ffn_mixed_gemm_workspace + my_id * llama_working_buffer->context_ffn_mixed_gemm_ws_bytes;
    }
    else if (int8_mode == 2) {
        const int max_size                                 = std::max(hidden_units, inter_size);
        size_t    context_ffn_int8_gemm_ws_bytes           = template_int8_fc_runner->getWorkspaceSize(token_num, max_size, max_size);
        llama_preprocessing->ffn_layer_->int8_gemm_workspace_    = llama_working_buffer->context_ffn_int8_gemm_workspace + my_id * llama_working_buffer->context_ffn_int8_gemm_ws_bytes;
    }
    

    // shared
    const size_t max_cache_seq_len             = total_seq_len;
    const size_t max_prefix_soft_prompt_length = fixed_prompt_len;
    size_t       max_input_len                 = fixed_input_len + max_prefix_soft_prompt_length;

    const size_t batchxbeam                    = working_batch_size;
    const size_t self_cache_size               = (decoder_layers / pipeline_para_size) * batchxbeam * max_cache_seq_len * local_hidden_units;

    int          local_vacab_size              = std::ceil(vocab_size / 1.f / tensor_para_size);
    if (std::is_same<half, T>::value) {
                 local_vacab_size              = std::ceil(local_vacab_size / 8.f) * 8;
    }
    size_t       vocab_size_padded             = (size_t)local_vacab_size * tensor_para_size;
    
    if (shared_contexts_ratio > 0.0f) {
        llama_preprocessing->shared_contexts_idx_   = llama_working_buffer->shared_contexts_idx_ + my_id * max_batch_size;
        llama_preprocessing->compact_idx_           = llama_working_buffer->compact_idx_+ my_id * batchxbeam;
        llama_preprocessing->batch_to_compact_idx_  = llama_working_buffer->batch_to_compact_idx_ + my_id * max_batch_size;
        llama_preprocessing->compact_size_          = llama_working_buffer->compact_size_;
    }

    if (vocab_size != vocab_size_padded) {
        llama_preprocessing->padded_embedding_kernel_               = llama_working_buffer->padded_embedding_kernel_;
        llama_preprocessing->padded_embedding_kernel_ptr_           = llama_preprocessing->padded_embedding_kernel_;
        llama_preprocessing->padded_embedding_bias_                 = llama_working_buffer->padded_embedding_bias_;
        llama_preprocessing->padded_embedding_bias_ptr_             = llama_preprocessing->padded_embedding_bias_;
    }
    
    llama_preprocessing->input_attention_mask_                      = llama_working_buffer->input_attention_mask_ + my_id * batchxbeam * total_seq_len * max_cache_seq_len;
    llama_preprocessing->decoder_input_buf_                         = llama_working_buffer->decoder_input_buf_ + my_id * batchxbeam * hidden_units;
    llama_preprocessing->decoder_output_buf_                        = llama_working_buffer->decoder_output_buf_ + my_id * batchxbeam * hidden_units;
    llama_preprocessing->normed_decoder_output_buf_                 = llama_working_buffer->normed_decoder_output_buf_ + my_id * batchxbeam * hidden_units;

    llama_preprocessing->logits_buf_                                = llama_working_buffer->logits_buf_ + my_id * batchxbeam * vocab_size_padded;
    llama_preprocessing->nccl_logits_buf_                           = llama_working_buffer->nccl_logits_buf_ + my_id * batchxbeam * vocab_size_padded;
    llama_preprocessing->cum_log_probs_                             = llama_working_buffer->cum_log_probs_ + my_id * batchxbeam;
    llama_preprocessing->sequence_lengths_                          = llama_working_buffer->sequence_lengths_ + my_id * batchxbeam;

    llama_preprocessing->finished_buf_                              = llama_working_buffer->finished_buf_ + my_id * batchxbeam;
    llama_preprocessing->h_finished_buf_                            = llama_working_buffer->h_finished_buf_ + my_id * batchxbeam;

    llama_preprocessing->key_cache_                                 = llama_working_buffer->key_cache_ + my_id * self_cache_size;
    llama_preprocessing->value_cache_                               = llama_working_buffer->value_cache_ + my_id * self_cache_size;

    if (beam_width > 1) {
        llama_preprocessing->cache_indirections_[0]                 = llama_working_buffer->cache_indirections_[0] + my_id * batchxbeam * max_cache_seq_len;
        llama_preprocessing->cache_indirections_[1]                 = llama_working_buffer->cache_indirections_[1] + my_id * batchxbeam * max_cache_seq_len;
    }

    llama_preprocessing->tiled_total_padding_count_                 = llama_working_buffer->tiled_total_padding_count_ + my_id * batchxbeam;

    llama_preprocessing->prompt_learning_weight_batch_              = llama_working_buffer->prompt_learning_weight_batch_ + my_id * batchxbeam;
    llama_preprocessing->tiled_prompt_lengths_buf_                  = llama_working_buffer->tiled_prompt_lengths_buf_ + my_id * batchxbeam;

    llama_preprocessing->tiled_input_ids_buf_                       = llama_working_buffer->tiled_input_ids_buf_ + my_id * batchxbeam * max_input_len;
    llama_preprocessing->tiled_input_lengths_buf_                   = llama_working_buffer->tiled_input_lengths_buf_ + my_id * batchxbeam;
    llama_preprocessing->transposed_output_ids_buf_                 = llama_working_buffer->transposed_output_ids_buf_ + my_id * batchxbeam * total_seq_len;
    llama_preprocessing->output_ids_buf_                            = llama_working_buffer->output_ids_buf_ + my_id * batchxbeam * total_seq_len;
    llama_preprocessing->parent_ids_buf_                            = llama_working_buffer->parent_ids_buf_ + my_id * batchxbeam * total_seq_len;

    llama_preprocessing->start_ids_buf_                             = llama_working_buffer->start_ids_buf_ + my_id * max_batch_size;
    llama_preprocessing->end_ids_buf_                               = llama_working_buffer->end_ids_buf_ + my_id * max_batch_size;
    llama_preprocessing->seq_limit_len_                             = llama_working_buffer->seq_limit_len_ + my_id * max_batch_size;
    
    llama_preprocessing->context_decoder_input_buf_                 = llama_working_buffer->context_decoder_input_buf_ + my_id * batchxbeam * max_input_len * hidden_units;
    llama_preprocessing->context_decoder_output_buf_                = llama_working_buffer->context_decoder_output_buf_ + my_id * batchxbeam * max_input_len * hidden_units;
    llama_preprocessing->output_log_probs_buf_                      = llama_working_buffer->output_log_probs_buf_ + my_id * batchxbeam * total_seq_len;

    llama_preprocessing->generation_should_stop_                    = llama_working_buffer->generation_should_stop_;
    // fprintf(stdout, "%d , size asgn %d\n", my_id, batchxbeam * max_cache_seq_len);
    llama_preprocessing->masked_tokens_                             = llama_working_buffer->masked_tokens_ + my_id * batchxbeam * max_cache_seq_len;

    llama_preprocessing->run(output_tensors, input_tensors, llama_model_weights);
  }
}

template<typename T>
void Flover<T>::InferenceStream(int* stop_flag, int setDeviceNo, std::promise<void> &p)
{
  int device;
  check_cuda_error(cudaSetDevice(setDeviceNo));
  check_cuda_error(cudaGetDevice(&device));

  const std::string model_name           = reader.Get("model_specification", "model_type");
  if (model_name == "gptj_6b") {
      
    inference_stream                       = gptj_inference<T>(reader, flover_stream, flover_tensor_para, flover_pipeline_para, flover_cublas);
    const size_t      beam_width           = reader.GetInteger("model_specification", "beam_width");
    const size_t      vocab_size           = reader.GetInteger(model_name, "vocab_size");
    int               tensor_para_size     = reader.GetInteger("model_specification", "tensor_para_size");
    int          local_vacab_size          = std::ceil(vocab_size / 1.f / tensor_para_size);
    if (std::is_same<half, T>::value) {
                  local_vacab_size         = std::ceil(local_vacab_size / 8.f) * 8;
    }
    size_t       vocab_size_padded         = (size_t)local_vacab_size * tensor_para_size;
    const size_t head_num                  = reader.GetInteger(model_name, "head_num");
    const size_t size_per_head             = reader.GetInteger(model_name, "size_per_head");
    const size_t hidden_units              = head_num * size_per_head;

    DecoderSelfAttentionLayer<T>* _temp_                        = dynamic_cast<DecoderSelfAttentionLayer<T>*>(inference_stream->self_attention_layer_);
    _temp_->qkv_buf_                                            = working_buffer->inference_attention_qkv_buf_;
    _temp_->context_buf_                                        = working_buffer->inference_attention_context_buf_;
    
    inference_stream->ffn_layer_->inter_buf_                    = working_buffer->inference_ffn_inter_buf;

    inference_stream->decoder_normed_input_                     = working_buffer->inference_decoder_decoder_normed_input_;
    inference_stream->self_attn_output_                         = working_buffer->inference_decoder_self_attn_output_;
    inference_stream->ffn_output_                               = working_buffer->inference_decoder_ffn_output_;
    inference_stream->decoder_layer_output_                     = working_buffer->inference_decoder_decoder_layer_output_;
      
    inference_stream->input_attention_mask_                     = working_buffer->input_attention_mask_;
    inference_stream->decoder_input_buf_                        = working_buffer->decoder_input_buf_;
    inference_stream->decoder_output_buf_                       = working_buffer->decoder_output_buf_;
    inference_stream->normed_decoder_output_buf_                = working_buffer->normed_decoder_output_buf_;

    inference_stream->logits_buf_                               = working_buffer->logits_buf_;
    inference_stream->nccl_logits_buf_                          = working_buffer->nccl_logits_buf_;
    inference_stream->cum_log_probs_                            = working_buffer->cum_log_probs_;
    inference_stream->sequence_lengths_                         = working_buffer->sequence_lengths_;

    inference_stream->finished_buf_                             = working_buffer->finished_buf_;
    inference_stream->h_finished_buf_                           = working_buffer->h_finished_buf_;

    inference_stream->key_cache_                                = working_buffer->key_cache_;
    inference_stream->value_cache_                              = working_buffer->value_cache_;

    if (beam_width > 1) {
        inference_stream->cache_indirections_[0]                = working_buffer->cache_indirections_[0];
        inference_stream->cache_indirections_[1]                = working_buffer->cache_indirections_[1];
    }
    inference_stream->tiled_total_padding_count_                = working_buffer->tiled_total_padding_count_;

    inference_stream->prompt_learning_weight_batch_             = working_buffer->prompt_learning_weight_batch_;
    inference_stream->tiled_prompt_lengths_buf_                 = working_buffer->tiled_prompt_lengths_buf_;

    inference_stream->tiled_input_ids_buf_                      = working_buffer->tiled_input_ids_buf_;
    inference_stream->tiled_input_lengths_buf_                  = working_buffer->tiled_input_lengths_buf_;
    inference_stream->transposed_output_ids_buf_                = working_buffer->transposed_output_ids_buf_;
    inference_stream->output_ids_buf_                           = working_buffer->output_ids_buf_;
    inference_stream->parent_ids_buf_                           = working_buffer->parent_ids_buf_;

    inference_stream->start_ids_buf_                            = working_buffer->start_ids_buf_;
    inference_stream->end_ids_buf_                              = working_buffer->end_ids_buf_;
    inference_stream->seq_limit_len_                            = working_buffer->seq_limit_len_;

    inference_stream->context_decoder_input_buf_                = working_buffer->context_decoder_input_buf_;
    inference_stream->context_decoder_output_buf_               = working_buffer->context_decoder_output_buf_;
    inference_stream->output_log_probs_buf_                     = working_buffer->output_log_probs_buf_;

    inference_stream->generation_should_stop_                   = working_buffer->generation_should_stop_;
    inference_stream->masked_tokens_                            = working_buffer->masked_tokens_;

    // GptJWeight<T>* _temp_weights           = dynamic_cast<GptJWeight<T>*>(model_weights);
    if (vocab_size == vocab_size_padded) {
        inference_stream->padded_embedding_kernel_ptr_ = model_weights->post_decoder_embedding.kernel;
        inference_stream->padded_embedding_bias_ptr_   = model_weights->post_decoder_embedding.bias;
    }
    else {
        cudaMemcpyAsync(inference_stream->padded_embedding_kernel_,
                        model_weights->post_decoder_embedding.kernel,
                        sizeof(T) * vocab_size * hidden_units,
                        cudaMemcpyDeviceToDevice,
                        flover_stream);
        cudaMemcpyAsync(inference_stream->padded_embedding_bias_,
                        model_weights->post_decoder_embedding.bias,
                        sizeof(T) * vocab_size,
                        cudaMemcpyDeviceToDevice,
                        flover_stream);
        sync_check_cuda_error();
    }
    cudaStreamSynchronize(flover_stream);
    cudaDeviceSynchronize;
    inference_stream->run(stop_flag, reader, model_weights, &inque);
  }
  else if (model_name.substr(0, 5) == "llama") {
    llama_inference_stream                 = llama_init_inference<T>(reader, flover_stream, flover_tensor_para, flover_pipeline_para, flover_cublas);
    const size_t      beam_width           = reader.GetInteger("model_specification", "beam_width");
    const size_t      vocab_size           = reader.GetInteger(model_name, "vocab_size");
    const size_t      per_batch_size       = reader.GetInteger("model_specification", "per_batch_size");
    const size_t      fixed_input_len      = reader.GetInteger("model_specification", "fixed_input_len");
    int               tensor_para_size     = reader.GetInteger("model_specification", "tensor_para_size");
    int          local_vacab_size          = std::ceil(vocab_size / 1.f / tensor_para_size);
    if (std::is_same<half, T>::value) {
                  local_vacab_size         = std::ceil(local_vacab_size / 8.f) * 8;
    }
    size_t       vocab_size_padded         = (size_t)local_vacab_size * tensor_para_size;
    const size_t head_num                  = reader.GetInteger(model_name, "head_num");
    const size_t size_per_head             = reader.GetInteger(model_name, "size_per_head");
    const size_t hidden_units              = head_num * size_per_head;

    const float       shared_contexts_ratio = reader.GetFloat(model_name, "shared_contexts_ratio");
    bool              use_shared_contexts   = (shared_contexts_ratio > 0.0f) && (fixed_input_len >= 1) && (per_batch_size > 1);
    
    if (shared_contexts_ratio > 0.0f) {
        llama_inference_stream->shared_contexts_idx_   = llama_working_buffer->shared_contexts_idx_;
        llama_inference_stream->compact_idx_           = llama_working_buffer->compact_idx_;
        llama_inference_stream->batch_to_compact_idx_  = llama_working_buffer->batch_to_compact_idx_;
        llama_inference_stream->compact_size_          = llama_working_buffer->compact_size_;
    }

    DecoderSelfAttentionLayer<T>* _temp_                        = dynamic_cast<DecoderSelfAttentionLayer<T>*>(llama_inference_stream->self_attention_layer_);
    _temp_->qkv_buf_                                            = llama_working_buffer->inference_attention_qkv_buf_;
    _temp_->context_buf_                                        = llama_working_buffer->inference_attention_context_buf_;
    
    llama_inference_stream->ffn_layer_->inter_buf_                    = llama_working_buffer->inference_ffn_inter_buf;
    llama_inference_stream->ffn_layer_->inter_buf_2_                  = llama_working_buffer->inference_ffn_inter_buf_2;

    llama_inference_stream->decoder_normed_input_                     = llama_working_buffer->inference_decoder_decoder_normed_input_;
    llama_inference_stream->self_attn_output_                         = llama_working_buffer->inference_decoder_self_attn_output_;
    llama_inference_stream->ffn_output_                               = llama_working_buffer->inference_decoder_ffn_output_;
    llama_inference_stream->decoder_layer_output_                     = llama_working_buffer->inference_decoder_decoder_layer_output_;
      
    llama_inference_stream->input_attention_mask_                     = llama_working_buffer->input_attention_mask_;
    llama_inference_stream->decoder_input_buf_                        = llama_working_buffer->decoder_input_buf_;
    llama_inference_stream->decoder_output_buf_                       = llama_working_buffer->decoder_output_buf_;
    llama_inference_stream->normed_decoder_output_buf_                = llama_working_buffer->normed_decoder_output_buf_;

    llama_inference_stream->logits_buf_                               = llama_working_buffer->logits_buf_;
    llama_inference_stream->nccl_logits_buf_                          = llama_working_buffer->nccl_logits_buf_;
    llama_inference_stream->cum_log_probs_                            = llama_working_buffer->cum_log_probs_;
    llama_inference_stream->sequence_lengths_                         = llama_working_buffer->sequence_lengths_;

    llama_inference_stream->finished_buf_                             = llama_working_buffer->finished_buf_;
    llama_inference_stream->h_finished_buf_                           = llama_working_buffer->h_finished_buf_;

    llama_inference_stream->key_cache_                                = llama_working_buffer->key_cache_;
    llama_inference_stream->value_cache_                              = llama_working_buffer->value_cache_;

    if (beam_width > 1) {
        llama_inference_stream->cache_indirections_[0]                = llama_working_buffer->cache_indirections_[0];
        llama_inference_stream->cache_indirections_[1]                = llama_working_buffer->cache_indirections_[1];
    }
    llama_inference_stream->tiled_total_padding_count_                = llama_working_buffer->tiled_total_padding_count_;

    llama_inference_stream->prompt_learning_weight_batch_             = llama_working_buffer->prompt_learning_weight_batch_;
    llama_inference_stream->tiled_prompt_lengths_buf_                 = llama_working_buffer->tiled_prompt_lengths_buf_;

    llama_inference_stream->tiled_input_ids_buf_                      = llama_working_buffer->tiled_input_ids_buf_;
    llama_inference_stream->tiled_input_lengths_buf_                  = llama_working_buffer->tiled_input_lengths_buf_;
    llama_inference_stream->transposed_output_ids_buf_                = llama_working_buffer->transposed_output_ids_buf_;
    llama_inference_stream->output_ids_buf_                           = llama_working_buffer->output_ids_buf_;
    llama_inference_stream->parent_ids_buf_                           = llama_working_buffer->parent_ids_buf_;

    llama_inference_stream->start_ids_buf_                            = llama_working_buffer->start_ids_buf_;
    llama_inference_stream->end_ids_buf_                              = llama_working_buffer->end_ids_buf_;
    llama_inference_stream->seq_limit_len_                            = llama_working_buffer->seq_limit_len_;

    llama_inference_stream->context_decoder_input_buf_                = llama_working_buffer->context_decoder_input_buf_;
    llama_inference_stream->context_decoder_output_buf_               = llama_working_buffer->context_decoder_output_buf_;
    llama_inference_stream->output_log_probs_buf_                     = llama_working_buffer->output_log_probs_buf_;

    llama_inference_stream->generation_should_stop_                   = llama_working_buffer->generation_should_stop_;
    llama_inference_stream->masked_tokens_                            = llama_working_buffer->masked_tokens_;

    llama_inference_stream->dynamic_decode_layer_->h_pinned_finished_sum_ = llama_working_buffer->dynamic_decoder_layer_h_pinned_finished_sum_;

    llama_inference_stream->dynamic_decode_layer_->topk_decode_->curandstate_buf_        = llama_working_buffer->topk_curandstate_buf_;
    llama_inference_stream->dynamic_decode_layer_->topk_decode_->random_seeds_buf_       = llama_working_buffer->topk_random_seeds_buf_;
    llama_inference_stream->dynamic_decode_layer_->topk_decode_->temperature_buf_        = llama_working_buffer->topk_temperature_buf_;
    llama_inference_stream->dynamic_decode_layer_->topk_decode_->repetition_penalty_buf_ = llama_working_buffer->topk_repetition_penalty_buf_;
    llama_inference_stream->dynamic_decode_layer_->topk_decode_->min_lengths_buf_        = llama_working_buffer->topk_min_lengths_buf_;
    llama_inference_stream->dynamic_decode_layer_->topk_decode_->runtime_logits_buf_     = llama_working_buffer->topk_runtime_logits_buf_;
    llama_inference_stream->dynamic_decode_layer_->topk_decode_->skip_decode_buf_        = llama_working_buffer->topk_skip_decode_buf_;
    llama_inference_stream->dynamic_decode_layer_->topk_decode_->sampling_workspace_size_= llama_working_buffer->topk_sampling_workspace_size_;
    llama_inference_stream->dynamic_decode_layer_->topk_decode_->sampling_workspace_     = llama_working_buffer->topk_sampling_workspace_;
    llama_inference_stream->dynamic_decode_layer_->topk_decode_->runtime_top_k_buf_      = llama_working_buffer->topk_runtime_top_k_buf_;
    llama_inference_stream->dynamic_decode_layer_->topk_decode_->runtime_top_p_buf_      = llama_working_buffer->topk_runtime_top_p_buf_;

    llama_inference_stream->dynamic_decode_layer_->topp_decode_->curandstate_buf_        = llama_working_buffer->topp_curandstate_buf_;
    llama_inference_stream->dynamic_decode_layer_->topp_decode_->random_seeds_buf_       = llama_working_buffer->topp_random_seeds_buf_;
    llama_inference_stream->dynamic_decode_layer_->topp_decode_->temperature_buf_        = llama_working_buffer->topp_temperature_buf_;
    llama_inference_stream->dynamic_decode_layer_->topp_decode_->repetition_penalty_buf_ = llama_working_buffer->topp_repetition_penalty_buf_;
    llama_inference_stream->dynamic_decode_layer_->topp_decode_->min_lengths_buf_        = llama_working_buffer->topp_min_lengths_buf_;
    llama_inference_stream->dynamic_decode_layer_->topp_decode_->runtime_logits_buf_     = llama_working_buffer->topp_runtime_logits_buf_;
    llama_inference_stream->dynamic_decode_layer_->topp_decode_->skip_decode_buf_        = llama_working_buffer->topp_skip_decode_buf_;
    llama_inference_stream->dynamic_decode_layer_->topp_decode_->sampling_workspace_size_= llama_working_buffer->topp_sampling_workspace_size_;
    llama_inference_stream->dynamic_decode_layer_->topp_decode_->sampling_workspace_     = llama_working_buffer->topp_sampling_workspace_;
    llama_inference_stream->dynamic_decode_layer_->topp_decode_->runtime_top_k_buf_      = llama_working_buffer->topp_runtime_top_k_buf_;
    llama_inference_stream->dynamic_decode_layer_->topp_decode_->runtime_top_p_buf_      = llama_working_buffer->topp_runtime_top_p_buf_;

    llama_inference_stream->dynamic_decode_layer_->topp_decode_->cub_temp_storage_size_  = llama_working_buffer->topp_cub_temp_storage_size_;
    llama_inference_stream->dynamic_decode_layer_->topp_decode_->topp_id_vals_buf_       = llama_working_buffer->topp_topp_id_vals_buf_;
    llama_inference_stream->dynamic_decode_layer_->topp_decode_->topp_offset_buf_        = llama_working_buffer->topp_topp_offset_buf_;
    llama_inference_stream->dynamic_decode_layer_->topp_decode_->begin_topp_offset_buf_  = llama_working_buffer->topp_begin_topp_offset_buf_;
    llama_inference_stream->dynamic_decode_layer_->topp_decode_->initial_top_p_buf_      = llama_working_buffer->topp_initial_top_p_buf_;
    llama_inference_stream->dynamic_decode_layer_->topp_decode_->top_p_decay_buf_        = llama_working_buffer->topp_top_p_decay_buf_;
    llama_inference_stream->dynamic_decode_layer_->topp_decode_->top_p_min_buf_          = llama_working_buffer->topp_top_p_min_buf_;
    llama_inference_stream->dynamic_decode_layer_->topp_decode_->top_p_reset_ids_buf_    = llama_working_buffer->topp_top_p_reset_ids_buf_;



    if (vocab_size == vocab_size_padded) {
        llama_inference_stream->padded_embedding_kernel_ptr_ = llama_model_weights->post_decoder_embedding.kernel;
        llama_inference_stream->padded_embedding_bias_ptr_   = llama_model_weights->post_decoder_embedding.bias;
    }
    else {
        cudaMemcpyAsync(llama_inference_stream->padded_embedding_kernel_,
                        llama_model_weights->post_decoder_embedding.kernel,
                        sizeof(T) * vocab_size * hidden_units,
                        cudaMemcpyDeviceToDevice,
                        flover_stream);
        cudaMemcpyAsync(llama_inference_stream->padded_embedding_bias_,
                        llama_model_weights->post_decoder_embedding.bias,
                        sizeof(T) * vocab_size,
                        cudaMemcpyDeviceToDevice,
                        flover_stream);
        sync_check_cuda_error();
    }
    cudaStreamSynchronize(flover_stream);
    cudaDeviceSynchronize;
    int use_mem_shuffle = reader.GetInteger("runtime_hyperparameter", "use_mem_shuffle");
    llama_inference_stream->run(stop_flag, reader, llama_model_weights, &inque, &this->new_req, &this->added, use_mem_shuffle);
  }
  p.set_value();
}


template<typename T>
void Flover<T>::addRequest(int defined_generation_len) {
  
  // log("Rank ", flover_tensor_para.rank_, " adding request...");
  reque.add_dummy(reader, task_id, defined_generation_len, 0);
  task_id++;
  // log("Rank ", flover_tensor_para.rank_, " adding request ", task_id - 1, " done");

}


template<typename T>
void Flover<T>::init_cublas() {
  cublasHandle_t   cublas_handle;
  cublasLtHandle_t cublaslt_handle;
  cublasCreate(&cublas_handle);
  cublasLtCreate(&cublaslt_handle);
  cublasSetStream(cublas_handle, flover_stream);
  cublasAlgoMap* cublas_algo_map = new cublasAlgoMap("gemm_config.in");

  std::mutex*     cublas_wrapper_mutex = new std::mutex();
  flover_cublas = new cublasMMWrapper(cublas_handle, cublaslt_handle, flover_stream, cublas_algo_map, cublas_wrapper_mutex, nullptr);
  if (std::is_same<T, half>::value) {
      flover_cublas->setGemmConfig(CUDA_R_16F, CUDA_R_16F, CUDA_R_16F, CUDA_R_32F);
  }
#ifdef ENABLE_BF16
  else if (std::is_same<T, __nv_bfloat16>::value) {
      flover_cublas->setBF16GemmConfig();
  }
#endif
  else if (std::is_same<T, float>::value) {
      flover_cublas->setFP32GemmConfig();
  }
}


template<typename T>
void Flover<T>::initialize() {
  print_mem_usage("Before initialization");
  const std::string model_type = reader.Get("model_specification", "model_type");
  const int tensor_para_size = reader.GetInteger("model_specification", "tensor_para_size");
  const int pipeline_para_size = reader.GetInteger("model_specification", "pipeline_para_size");
  flover_model_name = model_type;
  flover_max_concurrency = reader.GetInteger("model_specification", "max_concurrency");

  task_id = 0;
  ftNcclInitialize(flover_tensor_para, flover_pipeline_para, tensor_para_size, pipeline_para_size);
  mpi::barrier();

  for (int i=0;i<flover_max_concurrency;++i) {
    NcclParam _tensor_para;
    NcclParam _pipeline_para;
    ftNcclInitialize(_tensor_para, _pipeline_para, tensor_para_size, pipeline_para_size);
    sub_tensor_para.emplace_back(_tensor_para);
    sub_pipeline_para.emplace_back(_pipeline_para);
    mpi::barrier();
  }

  cudaStreamCreate(&flover_stream);

  init_cublas();

  if(model_type == "gptj_6b") {
    working_buffer = gptj_allocate_memory<T>(reader, flover_allocator);
    print_mem_usage("After allocating buffer");
    flover_cublas->cublas_workspace_    = working_buffer->cublas_workspace;

    model_weights = gptj_model_weights<T>(reader, flover_stream, flover_tensor_para, flover_pipeline_para, flover_cublas);
    print_mem_usage("After loading weights");
    // preprocessing = gptj_preprocessing<T>(reader, flover_stream, flover_tensor_para, flover_pipeline_para, flover_cublas);
    // print_mem_usage("After initializing preprocess");
  }
  else if (model_type.substr(0, 5) == "llama") {
    llama_working_buffer = llama_allocate_memory<T>(reader, flover_allocator, flover_stream);
    print_mem_usage("After allocating buffer");
    flover_cublas->cublas_workspace_    = llama_working_buffer->cublas_workspace;

    llama_model_weights = llama_init_model_weights<T>(reader, flover_stream, flover_tensor_para, flover_pipeline_para, flover_cublas);
    print_mem_usage("After loading weights");
  }
  cudaStreamSynchronize(flover_stream);
  cudaDeviceSynchronize;
}

template<typename T>
void Flover<T>::freeBuffer()
{
  if (flover_model_name == "gptj_6b")
    working_buffer->free(flover_allocator, flover_max_concurrency);
  else if (flover_model_name.substr(0, 5) == "llama")
    llama_working_buffer->free(flover_allocator, flover_max_concurrency);
}

template class Flover<float>;
template class Flover<half>;
#ifdef ENABLE_BF16
template class Flover<__nv_bfloat16>;
#endif

}