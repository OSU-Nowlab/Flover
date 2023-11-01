#pragma once
#include "src/flover/utils/allocator.h"

namespace flover {

template<typename T>
struct GptJBuffer {
    void*     cublas_workspace                       = nullptr;

    std::vector<int*>* coll_d_bad_words;
    std::vector<int*>* coll_d_stop_words;
    std::vector<int*>* coll_d_input_ids;
    std::vector<int*>* coll_d_input_lengths;
    std::vector<int*>* coll_d_output_ids;
    std::vector<int*>* coll_d_sequence_lengths;

    // preprocessing
    T*        context_decoder_decoder_normed_input   = nullptr;
    T*        context_decoder_self_attn_output       = nullptr;
    T*        context_decoder_ffn_output             = nullptr;
    T*        context_decoder_decoder_layer_output   = nullptr;
    size_t*   context_decoder_h_pinned_token_num_ptr = nullptr;
    int*      context_decoder_padding_offset         = nullptr;
    int*      context_decoder_cu_seqlens             = nullptr;
    uint32_t* context_decoder_seq_limit_len          = nullptr;

    T*        context_attention_qkv_buf              = nullptr;
    T*        context_attention_q_buf_2              = nullptr;
    T*        context_attention_k_buf_2              = nullptr;
    T*        context_attention_v_buf_2              = nullptr;
    T*        context_attention_qk_buf               = nullptr;
    float*    context_attention_qk_buf_float         = nullptr;
    T*        context_attention_qkv_buf_2            = nullptr;
    T*        context_attention_qkv_buf_3            = nullptr;
    char*     context_attention_mixed_gemm_workspace = nullptr;
    size_t    context_attention_mixed_gemm_ws_bytes  = 0;
    char*     context_attention_int8_gemm_workspace  = nullptr;
    size_t    context_attention_int8_gemm_ws_bytes   = 0;

    T*        context_ffn_inter_buf                  = nullptr;
    T*        context_ffn_inter_buf_2                = nullptr;  // for gated activation
    char*     context_ffn_mixed_gemm_workspace       = nullptr;
    size_t    context_ffn_mixed_gemm_ws_bytes        = 0;
    char*     context_ffn_int8_gemm_workspace        = nullptr;
    size_t    context_ffn_int8_gemm_ws_bytes         = 0;

    // inference
    T*        inference_decoder_decoder_normed_input_                  = nullptr;
    T*        inference_decoder_self_attn_output_                      = nullptr;
    T*        inference_decoder_ffn_output_                            = nullptr;
    T*        inference_decoder_decoder_layer_output_                  = nullptr;

    T*        inference_attention_qkv_buf_                             = nullptr;
    T*        inference_attention_context_buf_                         = nullptr;
    char*     inference_attention_mixed_gemm_workspace_                = nullptr;
    size_t    inference_attention_mixed_gemm_ws_bytes_                 = 0;
    char*     inference_attention_int8_gemm_workspace_                 = nullptr;
    size_t    inference_attention_int8_gemm_ws_bytes_                  = 0;
   
    T*        inference_ffn_inter_buf                  = nullptr;
    T*        inference_ffn_inter_buf_2                = nullptr;  // for gated activation
    char*     inference_ffn_mixed_gemm_workspace       = nullptr;
    size_t    inference_ffn_mixed_gemm_ws_bytes        = 0;
    char*     inference_ffn_int8_gemm_workspace        = nullptr;
    size_t    inference_ffn_int8_gemm_ws_bytes         = 0;


    // shared
    T*        padded_embedding_kernel_;
    T*        padded_embedding_bias_;
    const T*  padded_embedding_kernel_ptr_;
    const T*  padded_embedding_bias_ptr_;

    T*        input_attention_mask_;

    T*        decoder_input_buf_;
    T*        decoder_output_buf_;
    T*        normed_decoder_output_buf_;

    float*    logits_buf_;
    float*    nccl_logits_buf_;
    float*    cum_log_probs_;

    bool*     finished_buf_;
    bool*     h_finished_buf_;

    T*        key_cache_;
    T*        value_cache_;
    int*      cache_indirections_[2] = {nullptr, nullptr};

    // prompt_learning weight_batch ptrs
    const T** prompt_learning_weight_batch_;
    int*      tiled_prompt_lengths_buf_;  // only needed by prefix prompts

    int*      tiled_input_ids_buf_;
    int*      tiled_input_lengths_buf_;
    int*      transposed_output_ids_buf_;
    int*      output_ids_buf_;
    int*      parent_ids_buf_;
    int*      start_ids_buf_;
    int*      end_ids_buf_;
    uint32_t* seq_limit_len_             = nullptr;
    int*      tiled_total_padding_count_ = nullptr;

    bool*     generation_should_stop_    = nullptr;

    T*        context_decoder_input_buf_;
    T*        context_decoder_output_buf_;
    float*    output_log_probs_buf_;

    // function pointer callback
    // using         callback_sig                 = void(std::unordered_map<std::string, Tensor>*, void*);
    // callback_sig* token_generated_cb_          = nullptr;
    // void*         token_generated_ctx_         = nullptr;

    /* no alloc */
    bool*     masked_tokens_                   = nullptr;
    int*      sequence_lengths_                = nullptr;

    void free(IAllocator* allocator, const int max_concurrency)
    {

        for (int i = 0; i < max_concurrency; ++i) {
            cudaFree(coll_d_bad_words->at(i));
            cudaFree(coll_d_stop_words->at(i));
            cudaFree(coll_d_input_ids->at(i));
            cudaFree(coll_d_input_lengths->at(i));
            cudaFree(coll_d_output_ids->at(i));
            cudaFree(coll_d_sequence_lengths->at(i));
        }

        allocator->free((void**)(&context_decoder_decoder_normed_input));
        allocator->free((void**)(&context_decoder_self_attn_output));
        allocator->free((void**)(&context_decoder_ffn_output));
        allocator->free((void**)(&context_decoder_decoder_layer_output));
        allocator->free((void**)(&context_decoder_h_pinned_token_num_ptr), true);
        allocator->free((void**)(&context_decoder_padding_offset));
        allocator->free((void**)(&context_decoder_cu_seqlens));

        allocator->free((void**)(&context_attention_qkv_buf));
        allocator->free((void**)(&context_attention_q_buf_2));
        allocator->free((void**)(&context_attention_k_buf_2));
        allocator->free((void**)(&context_attention_v_buf_2));
        allocator->free((void**)(&context_attention_qk_buf));
        allocator->free((void**)(&context_attention_qk_buf_float));
        allocator->free((void**)(&context_attention_qkv_buf_2));
        allocator->free((void**)(&context_attention_qkv_buf_3));
        // allocator->free((void**)(&context_attention_mixed_gemm_workspace));
        // allocator->free((void**)(&context_attention_mixed_gemm_ws_bytes));
        // allocator->free((void**)(&context_attention_int8_gemm_workspace));
        // allocator->free((void**)(&context_attention_int8_gemm_ws_bytes));
        
        allocator->free((void**)(&context_ffn_inter_buf));
        // allocator->free((void**)(&context_ffn_inter_buf_2));
        // allocator->free((void**)(&context_ffn_mixed_gemm_workspace));
        // allocator->free((void**)(&context_ffn_mixed_gemm_ws_bytes));
        // allocator->free((void**)(&context_ffn_int8_gemm_workspace));
        // allocator->free((void**)(&context_ffn_int8_gemm_ws_bytes));

        allocator->free((void**)(&inference_decoder_decoder_normed_input_));
        allocator->free((void**)(&inference_decoder_self_attn_output_));
        allocator->free((void**)(&inference_decoder_ffn_output_));
        allocator->free((void**)(&inference_decoder_decoder_layer_output_));
        allocator->free((void**)(&inference_attention_qkv_buf_));
        allocator->free((void**)(&inference_attention_context_buf_));
        allocator->free((void**)(&inference_ffn_inter_buf));

        if (padded_embedding_kernel_ptr_ != nullptr) {
            padded_embedding_kernel_ptr_ = nullptr;
            padded_embedding_bias_ptr_   = nullptr;
            allocator->free((void**)(&padded_embedding_kernel_));
            allocator->free((void**)(&padded_embedding_bias_));
        }

        allocator->free((void**)(&input_attention_mask_));
        allocator->free((void**)(&decoder_input_buf_));
        allocator->free((void**)(&decoder_output_buf_));
        allocator->free((void**)(&normed_decoder_output_buf_));
        allocator->free((void**)(&logits_buf_));
        allocator->free((void**)(&nccl_logits_buf_));
        allocator->free((void**)(&cum_log_probs_));
        allocator->free((void**)(&sequence_lengths_));
        allocator->free((void**)(&finished_buf_));
        delete[] h_finished_buf_;

        allocator->free((void**)(&key_cache_));
        allocator->free((void**)(&value_cache_));
        if (cache_indirections_[0] != nullptr) {
            allocator->free((void**)(&cache_indirections_)[0]);
            allocator->free((void**)(&cache_indirections_)[1]);
        }

        allocator->free((void**)(&prompt_learning_weight_batch_));
        allocator->free((void**)(&tiled_prompt_lengths_buf_));
        allocator->free((void**)(&tiled_total_padding_count_));

        allocator->free((void**)(&tiled_input_ids_buf_));
        allocator->free((void**)(&tiled_input_lengths_buf_));

        allocator->free((void**)(&transposed_output_ids_buf_));
        allocator->free((void**)(&output_ids_buf_));
        allocator->free((void**)(&parent_ids_buf_));
        allocator->free((void**)(&masked_tokens_));

        allocator->free((void**)(&start_ids_buf_));
        allocator->free((void**)(&end_ids_buf_));
        allocator->free((void**)(&seq_limit_len_));

        allocator->free((void**)(&context_decoder_input_buf_));
        allocator->free((void**)(&context_decoder_output_buf_));
        allocator->free((void**)(&output_log_probs_buf_));

        allocator->free((void**)(&generation_should_stop_), true);
    }
};

}