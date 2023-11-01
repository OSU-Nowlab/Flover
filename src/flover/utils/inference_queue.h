#pragma once
#include "src/flover/utils/request_queue.h"
#include "src/flover/utils/Tensor.h"
#include <chrono>
#include <tbb/concurrent_queue.h>
#include <tbb/tbb.h>
#include <mutex>
#include <random>
#include <cstdlib>
#include <ctime>
#include <map>
#include <string>
#include <vector>

namespace flover {


struct InferenceInfo {
    int                cur_step;
    int                unique_task_id;
    int                mem_id;
    int                total_seq_len;

    int                per_batch_size;
    size_t             beam_width;
    uint               top_k;
    float              top_p;
    float              temperature;
    float              repetition_penalty;
    float              presence_penalty;
    float              len_penalty;
    int                min_length;
    size_t             request_batch_size;
    uint32_t           memory_len;
    int                prompt_learning_start_id;
    PromptLearningType prompt_learning_type;
    int                fixed_prompt_len;
    
    int                num_tasks;
    float              beam_search_diversity_rate;

    int                defined_generation_len;
    size_t             max_input_len;
    int                start_id;
    int                end_id;

    size_t             head_num;
    size_t             size_per_head;
    size_t             vocab_size;
    size_t             decoder_layers;
    size_t             rotary_embedding_dim;
    size_t             inter_size;
    size_t             hidden_units;
    size_t             tensor_para_size;
    size_t             pipeline_para_size;

    unsigned long long random_seed;

    int decoder_input_buf_len;
    int output_ids_buf_len;
    int tiled_total_padding_count_len;
    int finished_buf_len;
    int sequence_lengths_len;
    int tiled_prompt_lengths_buf_len;
    int cache_indirections_len;
    int masked_tokens_len;
    int decoder_output_buf_len;
    int key_cache_len;
    int value_cache_len;
    int start_ids_buf_len;
    int normed_decoder_output_buf_len;
    int nccl_logits_buf_len;

    int logits_buf_len;
    int tiled_input_lengths_buf_len;
    int end_ids_buf_len;
    int seq_limit_len_len;

    int cum_log_probs_len;
    int output_log_probs_buf_len;
    int parent_ids_buf_len;

    int decoder_layer_output_len;
    int decoder_normed_input_len;
    int self_attn_output_len;
    int ffn_output_len;

    long long start_time;

    std::unordered_map<std::string, Tensor>*       output_tensors;
    const std::unordered_map<std::string, Tensor>* input_tensors;


    
    int* d_bad_words;
    int* d_stop_words;
    int* d_input_ids;
    int* d_input_lengths;
    int* d_output_ids;
    int* d_sequence_lengths;
};

struct InferenceQueue {
    tbb::concurrent_queue<InferenceInfo> _queue;
    
    template<typename T>
    void add(const RequestInfo &value, 
            std::unordered_map<std::string, Tensor>*       output_tensors,
            const std::unordered_map<std::string, Tensor>* input_tensors,
            int* d_bad_words,
            int* d_stop_words,
            int* d_input_ids,
            int* d_input_lengths,
            int* d_output_ids,
            int* d_sequence_lengths) {

        InferenceInfo i;

        i.d_bad_words         = d_bad_words;
        i.d_stop_words        = d_stop_words;
        i.d_input_ids         = d_input_ids;
        i.d_input_lengths     = d_input_lengths;
        i.d_output_ids        = d_output_ids;
        i.d_sequence_lengths  = d_sequence_lengths;

        i.cur_step                   = value.max_input_len;
        i.unique_task_id             = value.unique_task_id;
        i.mem_id                     = value.unique_task_id;
        i.per_batch_size             = value.request_batch_size;
        i.beam_width                 = value.beam_width;
        i.top_k                      = value.top_k;
        i.top_p                      = value.top_p;
        i.temperature                = value.temperature;
        i.repetition_penalty         = value.repetition_penalty;
        i.presence_penalty           = value.presence_penalty;
        i.len_penalty                = value.len_penalty;
        i.min_length                 = value.min_length;
        i.request_batch_size         = value.request_batch_size;
        i.memory_len                 = value.memory_len;
        i.prompt_learning_start_id   = value.prompt_learning_start_id;
        i.prompt_learning_type       = value.prompt_learning_type;
        i.num_tasks                  = value.num_tasks;
        i.beam_search_diversity_rate = value.beam_search_diversity_rate;
        i.max_input_len              = value.max_input_len;
        i.start_id                   = value.start_id;
        i.end_id                     = value.end_id;
        i.random_seed                = value.random_seed;
        i.tensor_para_size           = value.tensor_para_size;
        i.pipeline_para_size         = value.pipeline_para_size;

        i.head_num                   = value.head_num;
        i.size_per_head              = value.size_per_head;
        i.vocab_size                 = value.vocab_size;
        i.decoder_layers             = value.decoder_layers;
        i.rotary_embedding_dim       = value.rotary_embedding_dim;
        i.inter_size                 = value.inter_size;
        i.hidden_units               = value.hidden_units;
        i.fixed_prompt_len           = value.fixed_prompt_len;

        int batchxbeam               = value.request_batch_size * value.beam_width;
        int hidden_units             = value.hidden_units;
        i.total_seq_len              = value.total_seq_len;
        int total_seq_len            = value.total_seq_len;
        size_t memory_len            = 0;
        size_t max_prefix_prompt_length = value.fixed_prompt_len;
        const size_t max_cache_seq_len = total_seq_len;
        const size_t self_cache_size = (value.decoder_layers / value.pipeline_para_size) * batchxbeam * max_cache_seq_len
                                   * hidden_units / value.tensor_para_size;

        i.decoder_input_buf_len         = batchxbeam * hidden_units;
        i.output_ids_buf_len            = batchxbeam * total_seq_len;
        i.tiled_total_padding_count_len = batchxbeam;
        i.finished_buf_len              = batchxbeam;
        i.sequence_lengths_len          = batchxbeam;
        i.tiled_prompt_lengths_buf_len  = batchxbeam;
        i.cache_indirections_len        = batchxbeam * max_cache_seq_len;
        i.masked_tokens_len             = batchxbeam * max_cache_seq_len;
        i.decoder_output_buf_len        = batchxbeam * hidden_units;
        i.key_cache_len                 = batchxbeam * max_cache_seq_len;
        i.value_cache_len               = batchxbeam * max_cache_seq_len;

        i.start_ids_buf_len             = value.request_batch_size;
        i.cum_log_probs_len             = batchxbeam;

        int          local_vacab_size          = std::ceil(value.vocab_size / 1.f / value.tensor_para_size);
        if (std::is_same<half, T>::value) {
                        local_vacab_size         = std::ceil(local_vacab_size / 8.f) * 8;
        }
        size_t       vocab_size_padded         = (size_t)local_vacab_size * value.tensor_para_size;

        i.normed_decoder_output_buf_len = batchxbeam * hidden_units;
        i.logits_buf_len                = batchxbeam * vocab_size_padded;
        i.nccl_logits_buf_len           = batchxbeam * vocab_size_padded;

        i.tiled_input_lengths_buf_len   = batchxbeam;
        i.end_ids_buf_len               = value.request_batch_size;
        i.seq_limit_len_len             = value.request_batch_size;

        i.output_ids_buf_len            = batchxbeam * max_cache_seq_len;
        i.finished_buf_len              = batchxbeam;
        i.cum_log_probs_len             = batchxbeam;
        i.output_log_probs_buf_len      = batchxbeam * max_cache_seq_len;
        i.parent_ids_buf_len            = batchxbeam * max_cache_seq_len;

        i.decoder_layer_output_len      = batchxbeam * hidden_units;
        i.decoder_normed_input_len      = batchxbeam * hidden_units;
        i.self_attn_output_len          = batchxbeam * hidden_units;
        i.ffn_output_len                = batchxbeam * hidden_units;

        i.output_tensors                = output_tensors;
        i.input_tensors                 = input_tensors;
        
        i.defined_generation_len        = value.defined_generation_len;
        i.start_time                    = value.start_time;
        _queue.push(i);
    }

    bool get(InferenceInfo &value) {
        return _queue.try_pop(value);
    }
};

}