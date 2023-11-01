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
#include "src/flover/models/gptj/gptj_preprocessing.h"
#include "src/flover/models/gptj/gptj_inference.h"
#include "src/flover/models/gptj/GptJDecoder.h"
#include "src/flover/models/gptj/GptJWeight.h"
#include "src/flover/utils/custom_ar_comm.h"
#include "src/flover/models/flover/buffers/gptj_buffers.h"
#include "src/flover/utils/word_list.h"
#include "src/flover/utils/nccl_utils.h"
#include "src/flover/utils/nvtx_utils.h"


namespace flover {

template<typename T>
GptJBuffer<T>* gptj_allocate_memory(const INIReader reader, IAllocator* allocator)
{
    GptJBuffer<T>*    working_buffer       = new GptJBuffer<T>();

    const std::string model_name           = reader.Get("model_specification", "model_type");
    const size_t      max_concurrency      = reader.GetInteger("model_specification", "max_concurrency");
    const size_t      per_batch_size       = reader.GetInteger("model_specification", "per_batch_size");
    const size_t      total_seq_len        = reader.GetInteger("model_specification", "max_seq_len");
    const size_t      fixed_input_len      = reader.GetInteger("model_specification", "fixed_input_len");
    const size_t      fixed_prompt_len     = reader.GetInteger("model_specification", "fixed_prompt_len");
    const size_t      beam_width           = reader.GetInteger("model_specification", "beam_width");
    int               tensor_para_size     = reader.GetInteger("model_specification", "tensor_para_size");
    int               pipeline_para_size   = reader.GetInteger("model_specification", "pipeline_para_size");
    const size_t      max_batch_size       = max_concurrency * per_batch_size;

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

    working_buffer->cublas_workspace                       = allocator->reMalloc(working_buffer->cublas_workspace, CUBLAS_WORKSPACE_SIZE * (1 + max_concurrency));


    working_buffer->coll_d_bad_words        = new std::vector<int*>(max_concurrency);
    working_buffer->coll_d_stop_words       = new std::vector<int*>(max_concurrency);
    working_buffer->coll_d_input_ids        = new std::vector<int*>(max_concurrency);
    working_buffer->coll_d_input_lengths    = new std::vector<int*>(max_concurrency);
    working_buffer->coll_d_output_ids       = new std::vector<int*>(max_concurrency);
    working_buffer->coll_d_sequence_lengths = new std::vector<int*>(max_concurrency);
    for (int i = 0; i < max_concurrency; ++i) {
        std::vector<int> stop_words;
        std::vector<int> bad_words;
        std::vector<int> tiled_stop_words;
        read_word_list("/home/yao.877/parallel_inference/projects/FasterTransformer_AR/FasterTransformer_MV2/examples/cpp/gptj/bad_words.csv", bad_words);
        read_word_list("/home/yao.877/parallel_inference/projects/FasterTransformer_AR/FasterTransformer_MV2/examples/cpp/gptj/stop_words.csv", stop_words);
        for (int j = 0; j < per_batch_size; j++) {
            tiled_stop_words.insert(tiled_stop_words.end(), stop_words.begin(), stop_words.end());
        }

        deviceMalloc(&(working_buffer->coll_d_bad_words->at(i)), bad_words.size(), false);
        deviceMalloc(&(working_buffer->coll_d_stop_words->at(i)), tiled_stop_words.size(), false);
        deviceMalloc(&(working_buffer->coll_d_input_ids->at(i)), per_batch_size * fixed_input_len, false);
        deviceMalloc(&(working_buffer->coll_d_input_lengths->at(i)), per_batch_size, false);
        
        deviceMalloc(&(working_buffer->coll_d_output_ids->at(i)), per_batch_size * beam_width * total_seq_len, false);
        deviceMalloc(&(working_buffer->coll_d_sequence_lengths->at(i)), per_batch_size * beam_width, false);
    }


    working_buffer->context_decoder_decoder_normed_input   = reinterpret_cast<T*>(allocator->reMalloc(working_buffer->context_decoder_decoder_normed_input, sizeof(T) * working_batch_size * context_working_len * hidden_units, false));
    working_buffer->context_decoder_self_attn_output       = reinterpret_cast<T*>(allocator->reMalloc(working_buffer->context_decoder_self_attn_output, sizeof(T) * working_batch_size * context_working_len * hidden_units, false));
    working_buffer->context_decoder_ffn_output             = reinterpret_cast<T*>(allocator->reMalloc(working_buffer->context_decoder_ffn_output, sizeof(T) * working_batch_size * context_working_len * hidden_units, false));
    working_buffer->context_decoder_decoder_layer_output   = reinterpret_cast<T*>(allocator->reMalloc(working_buffer->context_decoder_decoder_layer_output, sizeof(T) * working_batch_size * context_working_len * hidden_units, false));
    working_buffer->context_decoder_h_pinned_token_num_ptr = (size_t*)(allocator->reMalloc(working_buffer->context_decoder_h_pinned_token_num_ptr, sizeof(size_t) * max_concurrency, true, true));
    working_buffer->context_decoder_padding_offset         = reinterpret_cast<int*>(allocator->reMalloc(working_buffer->context_decoder_padding_offset, sizeof(int) * working_batch_size * context_working_len, false));
    working_buffer->context_decoder_cu_seqlens             = reinterpret_cast<int*>(allocator->reMalloc(working_buffer->context_decoder_cu_seqlens, sizeof(int) * (working_batch_size + working_batch_size), false));

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

    working_buffer->context_attention_qkv_buf              = (T*)(allocator->reMalloc(working_buffer->context_attention_qkv_buf, type_size * 3 * working_batch_size * context_working_len * local_hidden_units, true));
    working_buffer->context_attention_q_buf_2              = (T*)(allocator->reMalloc(working_buffer->context_attention_qkv_buf, sizeof(T) * working_batch_size * context_working_len * local_hidden_units, true));
    working_buffer->context_attention_k_buf_2              = (T*)(allocator->reMalloc(working_buffer->context_attention_qkv_buf, sizeof(T) * working_batch_size * context_working_len * local_hidden_units, true));
    working_buffer->context_attention_v_buf_2              = (T*)(allocator->reMalloc(working_buffer->context_attention_qkv_buf, sizeof(T) * working_batch_size * context_working_len * local_hidden_units, true));
    if (attn_flag) {
        working_buffer->context_attention_qk_buf           = (T*)(allocator->reMalloc(working_buffer->context_attention_qk_buf, sizeof(T) * working_batch_size * context_working_len * context_working_len * local_head_num, true));
    } else {
        allocator->free((void**)(&working_buffer->context_attention_qk_buf));
    }

    working_buffer->context_attention_qkv_buf_2            = (T*)(allocator->reMalloc(working_buffer->context_attention_qkv_buf_2, sizeof(T) * working_batch_size * context_working_len * local_hidden_units, true));
    working_buffer->context_attention_qkv_buf_3            = (T*)(allocator->reMalloc(working_buffer->context_attention_qkv_buf_2, type_size * working_batch_size * context_working_len * local_hidden_units, true));

    if (is_context_qk_buf_float) {
        if (attn_flag){
            working_buffer->context_attention_qk_buf_float = (float*)(allocator->reMalloc(working_buffer->context_attention_qk_buf_float, sizeof(float) * working_batch_size * context_working_len * context_working_len * local_head_num, true));
        }
        else {
            allocator->free((void**)(&working_buffer->context_attention_qk_buf_float));
        }
    }

    // used by each ctxt

    const auto type_size_1 = int8_mode == 2 ? sizeof(int8_t) : sizeof(T);
    const auto token_num   = working_batch_size * context_working_len;
    
    working_buffer->context_ffn_inter_buf                = (T*)(allocator->reMalloc(working_buffer->context_ffn_inter_buf, type_size_1 * token_num * inter_size, false));
    if (int8_mode == 1) {
    }
    else if (int8_mode == 2) {
    }

    // inference decoder
    working_buffer->inference_decoder_decoder_normed_input_   = reinterpret_cast<T*>(allocator->reMalloc(working_buffer->inference_decoder_decoder_normed_input_, sizeof(T) * working_batch_size * hidden_units, false));
    working_buffer->inference_decoder_self_attn_output_       = reinterpret_cast<T*>(allocator->reMalloc(working_buffer->inference_decoder_self_attn_output_, sizeof(T) * working_batch_size * hidden_units, false));
    working_buffer->inference_decoder_ffn_output_             = reinterpret_cast<T*>(allocator->reMalloc(working_buffer->inference_decoder_ffn_output_, sizeof(T) * working_batch_size * hidden_units, false));
    working_buffer->inference_decoder_decoder_layer_output_   = reinterpret_cast<T*>(allocator->reMalloc(working_buffer->inference_decoder_decoder_layer_output_, sizeof(T) * working_batch_size * hidden_units, false));
    
    working_buffer->inference_attention_qkv_buf_              = reinterpret_cast<T*>(allocator->reMalloc(working_buffer->inference_attention_qkv_buf_, type_size * working_batch_size * 3 * local_hidden_units, false));
    working_buffer->inference_attention_context_buf_          = reinterpret_cast<T*>(allocator->reMalloc(working_buffer->inference_attention_context_buf_, type_size * working_batch_size * local_hidden_units, false));
    // if (int8_mode == 1) {
    // }
    // else if (int8_mode == 2) {
    // }

    
    const auto token_num_1   = working_batch_size;
    working_buffer->inference_ffn_inter_buf                   = (T*)(allocator->reMalloc(working_buffer->inference_ffn_inter_buf, type_size_1 * token_num_1 * inter_size, false));
    // if (int8_mode == 1) {
    // }
    // else if (int8_mode == 2) {
    // }

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
        working_buffer->padded_embedding_kernel_     = (T*)(allocator->reMalloc(working_buffer->padded_embedding_kernel_, sizeof(T) * hidden_units * vocab_size_padded, true));
        working_buffer->padded_embedding_kernel_ptr_ = working_buffer->padded_embedding_kernel_;
        working_buffer->padded_embedding_bias_       = (T*)(allocator->reMalloc(working_buffer->padded_embedding_bias_, sizeof(T) * vocab_size_padded, true));
        working_buffer->padded_embedding_bias_ptr_   = working_buffer->padded_embedding_bias_;
    }
    
    working_buffer->input_attention_mask_      = (T*)(allocator->reMalloc(working_buffer->input_attention_mask_, sizeof(T) * batchxbeam * total_seq_len * max_cache_seq_len, false));
    working_buffer->decoder_input_buf_         = (T*)(allocator->reMalloc(working_buffer->decoder_input_buf_, sizeof(T) * batchxbeam * hidden_units, false));
    working_buffer->decoder_output_buf_        = (T*)(allocator->reMalloc(working_buffer->decoder_output_buf_, sizeof(T) * batchxbeam * hidden_units, false));
    working_buffer->normed_decoder_output_buf_ = (T*)(allocator->reMalloc(working_buffer->normed_decoder_output_buf_, sizeof(T) * batchxbeam * hidden_units, false));

    working_buffer->logits_buf_                = (float*)(allocator->reMalloc(working_buffer->logits_buf_, sizeof(float) * batchxbeam * vocab_size_padded, false));
    working_buffer->nccl_logits_buf_           = (float*)(allocator->reMalloc(working_buffer->nccl_logits_buf_, sizeof(float) * batchxbeam * vocab_size_padded, false));
    working_buffer->cum_log_probs_             = (float*)(allocator->reMalloc(working_buffer->cum_log_probs_, sizeof(float) * batchxbeam, false));
    working_buffer->sequence_lengths_          = (int*)(allocator->reMalloc(working_buffer->sequence_lengths_, sizeof(int) * batchxbeam, false));

    working_buffer->finished_buf_              = (bool*)(allocator->reMalloc(working_buffer->finished_buf_, sizeof(bool) * batchxbeam, false));
    working_buffer->h_finished_buf_            = new bool[batchxbeam];

    working_buffer->key_cache_                 = (T*)(allocator->reMalloc(working_buffer->key_cache_, sizeof(T) * self_cache_size, true));
    working_buffer->value_cache_               = (T*)(allocator->reMalloc(working_buffer->key_cache_, sizeof(T) * self_cache_size, true));

    if (beam_width > 1) {
        working_buffer->cache_indirections_[0] = (int*)(allocator->reMalloc(working_buffer->cache_indirections_[0], sizeof(int) * batchxbeam * max_cache_seq_len, true));
        working_buffer->cache_indirections_[1] = (int*)(allocator->reMalloc(working_buffer->cache_indirections_[1], sizeof(int) * batchxbeam * max_cache_seq_len, true));
    }

    working_buffer->tiled_total_padding_count_    = (int*)(allocator->reMalloc(working_buffer->tiled_total_padding_count_, batchxbeam * sizeof(int), false));

    working_buffer->prompt_learning_weight_batch_ = (const T**)(allocator->reMalloc(working_buffer->prompt_learning_weight_batch_, sizeof(T*) * batchxbeam, false));
    working_buffer->tiled_prompt_lengths_buf_     = (int*)(allocator->reMalloc(working_buffer->tiled_prompt_lengths_buf_, sizeof(int) * batchxbeam, false));

    working_buffer->tiled_input_ids_buf_          = (int*)(allocator->reMalloc(working_buffer->tiled_input_ids_buf_, sizeof(int) * batchxbeam * max_input_len, true));
    working_buffer->tiled_input_lengths_buf_      = (int*)(allocator->reMalloc(working_buffer->tiled_input_lengths_buf_, sizeof(int) * batchxbeam, true));
    working_buffer->transposed_output_ids_buf_    = (int*)(allocator->reMalloc(working_buffer->transposed_output_ids_buf_, sizeof(int) * batchxbeam * total_seq_len, true));
    working_buffer->output_ids_buf_               = (int*)(allocator->reMalloc(working_buffer->output_ids_buf_, sizeof(int) * batchxbeam * total_seq_len, true));
    working_buffer->parent_ids_buf_               = (int*)(allocator->reMalloc(working_buffer->parent_ids_buf_, sizeof(int) * batchxbeam * total_seq_len, true));

    working_buffer->start_ids_buf_                = (int*)(allocator->reMalloc(working_buffer->start_ids_buf_, sizeof(int) * max_batch_size, false));
    working_buffer->end_ids_buf_                  = (int*)(allocator->reMalloc(working_buffer->end_ids_buf_, sizeof(int) * max_batch_size, false));
    working_buffer->seq_limit_len_                = (uint32_t*)(allocator->reMalloc(working_buffer->seq_limit_len_, sizeof(uint32_t) * max_batch_size, false));

    working_buffer->context_decoder_input_buf_    = (T*)(allocator->reMalloc(working_buffer->context_decoder_input_buf_, sizeof(T) * batchxbeam * max_input_len * hidden_units, false));
    working_buffer->context_decoder_output_buf_   = (T*)(allocator->reMalloc(working_buffer->context_decoder_output_buf_, sizeof(T) * batchxbeam * max_input_len * hidden_units, false));

    working_buffer->output_log_probs_buf_         = (float*)(allocator->reMalloc(working_buffer->output_log_probs_buf_, sizeof(float) * batchxbeam * total_seq_len, false));

    working_buffer->generation_should_stop_       = (bool*)(allocator->reMalloc(working_buffer->generation_should_stop_, sizeof(bool), true, true));
    fprintf(stdout, "size alloc %d\n", batchxbeam * max_cache_seq_len);
    working_buffer->masked_tokens_                = (bool*)(allocator->reMalloc(working_buffer->masked_tokens_, sizeof(bool) * batchxbeam * max_cache_seq_len, true));
    // function pointer callback
    using         callback_sig                 = void(std::unordered_map<std::string, Tensor>*, void*);
    callback_sig* token_generated_cb_          = nullptr;
    void*         token_generated_ctx_         = nullptr;
    sync_check_cuda_error();
    return working_buffer;
}

template<typename T>
GptJWeight<T>* gptj_model_weights(const INIReader reader, cudaStream_t flover_stream, NcclParam flover_tensor_para, NcclParam flover_pipeline_para, cublasMMWrapper* flover_cublas_wrapper)
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
    std::string model_dir = std::string(reader.Get("model_specification", "model_dir"));

    int prompt_learning_start_id = reader.GetInteger(model_name, "prompt_learning_start_id", end_id + 1);
    PromptLearningType prompt_learning_type =
        static_cast<PromptLearningType>(reader.GetInteger(model_name, "prompt_learning_type", 0));

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

    model_dir = model_dir + "/" + std::to_string(flover_tensor_para.world_size_) + "-gpu/";
    flover::GptJWeight<T>* gpt_weights = new flover::GptJWeight<T>(
        hidden_units,
        inter_size,
        vocab_size,
        decoder_layers,
        total_seq_len,
        flover_tensor_para.world_size_,
        flover_tensor_para.rank_,
        flover_pipeline_para.world_size_,
        flover_pipeline_para.rank_,
        prompt_learning_type,
        prefix_prompt_table_pair);  // optional if you don't need prefix prompts

    fprintf(stdout, "%d %d %d %d\n", flover_tensor_para.world_size_, flover_tensor_para.rank_, flover_pipeline_para.world_size_, flover_pipeline_para.rank_);
    gpt_weights->loadModel(model_dir);
    return gpt_weights;
}

template<typename T>
GptJPreprocessing<T>* gptj_preprocessing(const INIReader reader, cudaStream_t stream, NcclParam tensor_para, NcclParam pipeline_para, cublasMMWrapper* cublas_wrapper)
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

    const bool                                             neox_rotary_style           = false;  // A unify way for GPT-NeoX in the future, not used now.
    static constexpr float                                 layernorm_eps                = 1e-6f;
    bool                                                   is_free_buffer_after_forward = false;
    const bool                                             is_context_qk_buf_float      = (std::getenv("CONTEXT_ATTENTION_BMM1_HALF_ACCUM") == nullptr || std::string(std::getenv("CONTEXT_ATTENTION_BMM1_HALF_ACCUM")) != "ON");
    
    std::shared_ptr<AbstractCustomComm>                    custom_all_reduce_comm   = nullptr;
    int                                                    enable_custom_all_reduce = 0;

    GptJPreprocessing<T>* gpt_context_decoder = new GptJPreprocessing<T>(0,
                                                     0,
                                                     head_num,
                                                     size_per_head,
                                                     inter_size,
                                                     decoder_layers,
                                                     rotary_embedding_dim,
                                                     neox_rotary_style,
                                                     layernorm_eps,
                                                     tensor_para,
                                                     pipeline_para,
                                                     stream,
                                                     cublas_wrapper,
                                                     is_free_buffer_after_forward,
                                                     is_context_qk_buf_float,
                                                     attention_type,
                                                     custom_all_reduce_comm,
                                                     enable_custom_all_reduce);
    
    return gpt_context_decoder;
}


template<typename T>
GptJInference<T>* gptj_inference(const INIReader reader, cudaStream_t stream, NcclParam tensor_para, NcclParam pipeline_para, cublasMMWrapper* cublas_wrapper)
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


    const bool                                             neox_rotary_style           = false;  // A unify way for GPT-NeoX in the future, not used now.
    static constexpr float                                 layernorm_eps                = 1e-6f;
    bool                                                   is_free_buffer_after_forward = false;
    std::shared_ptr<AbstractCustomComm>                    custom_all_reduce_comm   = nullptr;
    int                                                    enable_custom_all_reduce = 0;

    GptJInference<T>* gpt_inference_decoder = new GptJInference<T>(0,
                                                     head_num,
                                                     size_per_head,
                                                     inter_size,
                                                     decoder_layers,
                                                     vocab_size,
                                                     rotary_embedding_dim,
                                                     neox_rotary_style,
                                                     layernorm_eps,
                                                     tensor_para,
                                                     pipeline_para,
                                                     stream,
                                                     cublas_wrapper,
                                                     is_free_buffer_after_forward,
                                                     custom_all_reduce_comm,
                                                     enable_custom_all_reduce);
    
    return gpt_inference_decoder;
}

}
