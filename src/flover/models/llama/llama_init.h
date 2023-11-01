#pragma once

#include <cstddef>
#include <vector>
#include <tbb/concurrent_queue.h>
#include <tbb/tbb.h>
#include <thread>
#include <stack>
#include <atomic>
#include <limits>
#include <queue>
#include <map>
#include <numeric>
#include <algorithm>
#include <cmath>

#include "src/flover/layers/DynamicDecodeLayer.h"
#include "src/flover/models/llama/llama_preprocessing.h"
#include "src/flover/models/llama/llama_inference.h"
#include "src/flover/models/llama/LlamaWeight.h"
#include "src/flover/utils/custom_ar_comm.h"
#include "src/flover/models/flover/buffers/llama_buffers.h"
#include "src/flover/utils/word_list.h"
#include "src/flover/utils/nccl_utils.h"
#include "src/flover/utils/nvtx_utils.h"


namespace flover {

template<typename T>
LlamaBuffer<T>* llama_allocate_memory(const INIReader reader, IAllocator* allocator, cudaStream_t stream);


template<typename T>
LlamaWeight<T>* llama_init_model_weights(const INIReader reader, cudaStream_t flover_stream, NcclParam flover_tensor_para, NcclParam flover_pipeline_para, cublasMMWrapper* flover_cublas_wrapper)
{
    const std::string model_name           = reader.Get("model_specification", "model_type");
    const size_t      max_concurrency      = reader.GetInteger("model_specification", "max_concurrency");
    const size_t      per_batch_size       = reader.GetInteger("model_specification", "per_batch_size");
    const size_t      total_seq_len        = reader.GetInteger("model_specification", "max_seq_len");
    const size_t      beam_width           = reader.GetInteger("model_specification", "beam_width");
    int               tensor_para_size     = reader.GetInteger("model_specification", "tensor_para_size");
    const size_t      max_batch_size       = max_concurrency * per_batch_size;

    const size_t      head_num             = reader.GetInteger(model_name, "head_num");
    const size_t      size_per_head        = reader.GetInteger(model_name, "size_per_head");
    const size_t      vocab_size           = reader.GetInteger(model_name, "vocab_size");
    const size_t      decoder_layers       = reader.GetInteger(model_name, "decoder_layers");
    const size_t      rotary_embedding_dim = reader.GetInteger(model_name, "rotary_embedding");
    const size_t      inter_size           = reader.GetInteger(model_name, "inter_size");
    const size_t      hidden_units         = head_num * size_per_head;

    const uint        top_k              = (uint)reader.GetInteger("runtime_hyperparameter", "top_k");
    const float       top_p              = reader.GetFloat("runtime_hyperparameter", "top_p");
    const float       temperature        = reader.GetFloat("runtime_hyperparameter", "temperature");
    const float       repetition_penalty = reader.GetFloat("runtime_hyperparameter", "repetition_penalty", 1.0f);
    const float       presence_penalty   = reader.GetFloat("runtime_hyperparameter", "presence_penalty", 0.0f);
    const float       len_penalty        = reader.GetFloat("runtime_hyperparameter", "len_penalty");
    const int         min_length         = reader.GetInteger("runtime_hyperparameter", "min_length", 0);
    const int         start_id           = reader.GetInteger(model_name, "start_id");
    const int         end_id             = reader.GetInteger(model_name, "end_id");
    const float       beam_search_diversity_rate =
        reader.GetFloat("runtime_hyperparameter", "beam_search_diversity_rate");
    std::string model_dir = std::string(reader.Get(model_name, "model_dir"));

    int prompt_learning_start_id = reader.GetInteger(model_name, "prompt_learning_start_id", end_id + 1);
    PromptLearningType prompt_learning_type =
        static_cast<PromptLearningType>(reader.GetInteger(model_name, "prompt_learning_type", 0));

    // NOTE: specify task names, take name id, prompt length in order to load those prompt learning tables.
    // NOTE: Please make sure task ids are continuous and start from 0
    // for example:
    // std::map<std::string, std::pair<int, int>> prefix_prompt_table_pair{{"no_prompt", {0, 0}},
    //                                                                     {"prompt_1", {1, 1}},
    //                                                                     {"prompt_2", {2, 2}},
    //                                                                     {"prompt_3", {3, 3}},
    //                                                                     {"prompt_4", {4, 4}},
    //                                                                     {"prompt_5", {5, 5}}};

    std::map<std::string, std::pair<int, int>> prefix_prompt_table_pair;
    const int num_tasks = reader.GetInteger(model_name, "num_tasks", 0);
    for (int task_name_id = 0; task_name_id < num_tasks; task_name_id++) {
        std::string config_task_name = model_name + "_task_" + std::to_string(task_name_id);
        std::string task_name        = reader.Get(config_task_name, "task_name");
        const int   prompt_length    = reader.GetInteger(config_task_name, "prompt_length", 0);
        prefix_prompt_table_pair.insert({task_name, {task_name_id, prompt_length}});
    }

    int rank       = mpi::getCommWorldRank();
    int world_size = mpi::getCommWorldSize();
    int device, device_count;
    check_cuda_error(cudaGetDeviceCount(&device_count));
    check_cuda_error(cudaSetDevice(rank % device_count));
    check_cuda_error(cudaGetDevice(&device));

    struct cudaDeviceProp prop;
    check_cuda_error(cudaGetDeviceProperties(&prop, device));
    printf("Device %s\n", prop.name);

    printf("P%d is running with %d GPU.\n", rank, device);

    unsigned long long random_seed;
    if (rank == 0) {
        random_seed = (unsigned long long)(0);
    }
    if (world_size > 1) {
        mpi::bcast(&random_seed, 1, mpi::MPI_TYPE_UNSIGNED_LONG_LONG, 0, mpi::COMM_WORLD);
    }

    const bool use_gptj_residual = false;
    int        int8_mode         = 0;

    model_dir = model_dir + "/" + std::to_string(flover_tensor_para.world_size_) + "-gpu/";
    
    flover::LlamaWeight<T>* gpt_weights = new flover::LlamaWeight<T>(
        hidden_units,
        inter_size,
        vocab_size,
        decoder_layers,
        0,
        flover_tensor_para.world_size_,
        flover_tensor_para.rank_,
        flover_pipeline_para.world_size_,
        flover_pipeline_para.rank_,
        use_gptj_residual,
        int8_mode,
        prompt_learning_type,
        prefix_prompt_table_pair);  // optional if you don't need prefix prompts

    fprintf(stdout, "%d %d %d %d\n", flover_tensor_para.world_size_, flover_tensor_para.rank_, flover_pipeline_para.world_size_, flover_pipeline_para.rank_);
    gpt_weights->loadModel(model_dir);
    return gpt_weights;
}


template<typename T>
LlamaPreprocessing<T>* llama_init_preprocessing(const INIReader reader, cudaStream_t stream, NcclParam tensor_para, NcclParam pipeline_para, cublasMMWrapper* cublas_wrapper)
{
    const std::string model_name           = reader.Get("model_specification", "model_type");
    const size_t      max_concurrency      = reader.GetInteger("model_specification", "max_concurrency");
    const size_t      per_batch_size       = reader.GetInteger("model_specification", "per_batch_size");
    const size_t      beam_width           = reader.GetInteger("model_specification", "beam_width");
    const size_t      max_batch_size       = max_concurrency * per_batch_size;

    const size_t      head_num             = reader.GetInteger(model_name, "head_num");
    const size_t      size_per_head        = reader.GetInteger(model_name, "size_per_head");
    const size_t      vocab_size           = reader.GetInteger(model_name, "vocab_size");
    const size_t      decoder_layers       = reader.GetInteger(model_name, "decoder_layers");
    const size_t      rotary_embedding_dim = reader.GetInteger(model_name, "rotary_embedding");
    const size_t      inter_size           = reader.GetInteger(model_name, "inter_size");
    const size_t      hidden_units         = head_num * size_per_head;

    const uint        top_k              = (uint)reader.GetInteger("runtime_hyperparameter", "top_k");
    const float       top_p              = reader.GetFloat("runtime_hyperparameter", "top_p");
    const float       temperature        = reader.GetFloat("runtime_hyperparameter", "temperature");
    const float       repetition_penalty = reader.GetFloat("runtime_hyperparameter", "repetition_penalty", 1.0f);
    const float       presence_penalty   = reader.GetFloat("runtime_hyperparameter", "presence_penalty", 0.0f);
    const float       len_penalty        = reader.GetFloat("runtime_hyperparameter", "len_penalty");
    const int         min_length         = reader.GetInteger("runtime_hyperparameter", "min_length", 0);
    const int         start_id           = reader.GetInteger(model_name, "start_id");
    const int         end_id             = reader.GetInteger(model_name, "end_id");
    const float       beam_search_diversity_rate =
        reader.GetFloat("runtime_hyperparameter", "beam_search_diversity_rate");

    AttentionType attention_type = getAttentionType<T>(size_per_head,
                                    getSMVersion(),
                                    true,   // remove_padding
                                    0,      // gpt supports any-seq-length fmha
                                    true,   // is_fuse
                                    false,  // with_relative_position_bias
                                    true);  // causal_mask

    const bool                                             neox_rotary_style            = false;  // A unify way for GPT-NeoX in the future, not used now.
    static constexpr float                                 layernorm_eps                = 1e-6f;
    bool                                                   is_free_buffer_after_forward = false;
    const bool                                             is_context_qk_buf_float      = (std::getenv("CONTEXT_ATTENTION_BMM1_HALF_ACCUM") == nullptr || std::string(std::getenv("CONTEXT_ATTENTION_BMM1_HALF_ACCUM")) != "ON");
    
    std::shared_ptr<AbstractCustomComm>                    custom_all_reduce_comm       = nullptr;
    int                                                    enable_custom_all_reduce     = 0;

    const bool                                             use_gptj_residual            = false;
    int                                                    int8_mode                    = 0;
    LlamaPreprocessing<T>* gpt_context_decoder = new LlamaPreprocessing<T>(0,
                                                     0,
                                                     head_num,
                                                     size_per_head,
                                                     inter_size,
                                                     decoder_layers,
                                                     rotary_embedding_dim,
                                                     neox_rotary_style,
                                                     use_gptj_residual,
                                                     layernorm_eps,
                                                     tensor_para,
                                                     pipeline_para,
                                                     stream,
                                                     cublas_wrapper,
                                                     is_free_buffer_after_forward,
                                                     is_context_qk_buf_float,
                                                     attention_type,
                                                     int8_mode,
                                                     custom_all_reduce_comm,
                                                     enable_custom_all_reduce);
    
    return gpt_context_decoder;
}

template<typename T>
LlamaInference<T>* llama_init_inference(const INIReader reader, cudaStream_t stream, NcclParam tensor_para, NcclParam pipeline_para, cublasMMWrapper* cublas_wrapper)
{
    const std::string model_name           = reader.Get("model_specification", "model_type");
    const size_t      max_concurrency      = reader.GetInteger("model_specification", "max_concurrency");
    const size_t      per_batch_size       = reader.GetInteger("model_specification", "per_batch_size");
    const size_t      beam_width           = reader.GetInteger("model_specification", "beam_width");
    const size_t      max_batch_size       = max_concurrency * per_batch_size;

    const size_t      head_num             = reader.GetInteger(model_name, "head_num");
    const size_t      size_per_head        = reader.GetInteger(model_name, "size_per_head");
    const size_t      vocab_size           = reader.GetInteger(model_name, "vocab_size");
    const size_t      decoder_layers       = reader.GetInteger(model_name, "decoder_layers");
    const size_t      rotary_embedding_dim = reader.GetInteger(model_name, "rotary_embedding");
    const size_t      inter_size           = reader.GetInteger(model_name, "inter_size");
    const size_t      hidden_units         = head_num * size_per_head;

    const uint        top_k              = (uint)reader.GetInteger("runtime_hyperparameter", "top_k");
    const float       top_p              = reader.GetFloat("runtime_hyperparameter", "top_p");
    const float       temperature        = reader.GetFloat("runtime_hyperparameter", "temperature");
    const float       repetition_penalty = reader.GetFloat("runtime_hyperparameter", "repetition_penalty", 1.0f);
    const float       presence_penalty   = reader.GetFloat("runtime_hyperparameter", "presence_penalty", 0.0f);
    const float       len_penalty        = reader.GetFloat("runtime_hyperparameter", "len_penalty");
    const int         min_length         = reader.GetInteger("runtime_hyperparameter", "min_length", 0);
    const int         start_id           = reader.GetInteger(model_name, "start_id");
    const int         end_id             = reader.GetInteger(model_name, "end_id");
    const float       beam_search_diversity_rate =
        reader.GetFloat("runtime_hyperparameter", "beam_search_diversity_rate");


    const bool                                             neox_rotary_style            = false;  // A unify way for GPT-NeoX in the future, not used now.
    const bool                                             use_gptj_residual            = false;
    static constexpr float                                 layernorm_eps                = 1e-6f;
    bool                                                   is_free_buffer_after_forward = false;
    std::shared_ptr<AbstractCustomComm>                    custom_all_reduce_comm       = nullptr;
    int                                                    enable_custom_all_reduce     = 0;
    int                                                    int8_mode                    = reader.GetInteger("model_specification", "int8_mode", 0);

    LlamaInference<T>* gpt_inference_decoder = new LlamaInference<T>(0,
                                                     head_num,
                                                     size_per_head,
                                                     inter_size,
                                                     decoder_layers,
                                                     vocab_size,
                                                     rotary_embedding_dim,
                                                     neox_rotary_style,
                                                     use_gptj_residual,
                                                     layernorm_eps,
                                                     tensor_para,
                                                     pipeline_para,
                                                     stream,
                                                     cublas_wrapper,
                                                     is_free_buffer_after_forward,
                                                     int8_mode,
                                                     custom_all_reduce_comm,
                                                     enable_custom_all_reduce);
                                                     
    return gpt_inference_decoder;
}

}