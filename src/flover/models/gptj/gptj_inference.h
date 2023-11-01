/*
 * Copyright (c) 2019-2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <vector>

#include "src/flover/kernels/add_residual_kernels.h"
#include "src/flover/kernels/layernorm_kernels.h"
#include "src/flover/layers/BaseLayer.h"
#include "src/flover/layers/FfnLayer.h"
#include "src/flover/layers/attention_layers/BaseAttentionLayer.h"
#include "src/flover/models/gptj/GptJDecoderLayerWeight.h"
#include "src/flover/models/gptj/GptJWeight.h"
#include "src/flover/utils/Tensor.h"
#include "src/flover/utils/allocator.h"
#include "src/flover/utils/cublasMMWrapper.h"
#include "src/flover/utils/custom_ar_comm.h"
#include "src/flover/utils/nccl_utils.h"
#include "src/flover/layers/DynamicDecodeLayer.h"
#include "src/flover/models/gptj/GptJContextDecoder.h"
#include "src/flover/models/gptj/GptJDecoder.h"
#include "src/flover/utils/inference_queue.h"
#include "src/flover/utils/request_queue.h"
#include "src/flover/utils/flover_logger.h"

#include <atomic>

namespace flover {


template<typename T>
class GptJInference: public BaseLayer {
private:
    int actives = 0;
    int running_batchsize = 0;
    int decoder_input_buf_len = 0;
    int output_ids_buf_len = 0;
    int tiled_total_padding_count_len = 0;
    int finished_buf_len = 0;
    int sequence_lengths_len = 0;
    int tiled_prompt_lengths_buf_len = 0;
    int cache_indirections_len = 0;
    int masked_tokens_len = 0;
    int decoder_output_buf_len = 0;
    int key_cache_len = 0;
    int value_cache_len = 0;

    int decoder_input_buf_offset = 0;
    int output_ids_buf_offset = 0;
    int tiled_total_padding_count_offset = 0;
    int finished_buf_offset = 0;
    int sequence_lengths_offset = 0;
    int tiled_prompt_lengths_buf_offset = 0;
    int cache_indirections_offset = 0;
    int masked_tokens_offset = 0;
    int decoder_output_buf_offset = 0;
    int key_cache_offset = 0;
    int value_cache_offset = 0;

    std::unordered_map<int, std::unique_ptr<InferenceInfo>> InferenceStatus;

    int total_iters;

protected:
    void         allocateBuffer() override;
    void         freeBuffer() override;
    bool         isValidLayerParallelId(uint l);
    bool         isFirstLayerParallelId(uint l);
    bool         isLastLayerParallelId(uint l);
    int          getFirstLayerParallelId();
    virtual void initialize();
    // buffer handling
    size_t max_batch_size_ = 0;

    // meta data
    size_t vocab_size_;
    size_t head_num_;
    size_t size_per_head_;
    size_t inter_size_;
    size_t num_layer_;
    size_t rotary_embedding_dim_;
    bool   neox_rotary_style_;  // A unify way for GPT-NeoX in the future, not used now.
    size_t hidden_units_;

    float layernorm_eps_;

    NcclParam tensor_para_;
    NcclParam pipeline_para_;

    std::shared_ptr<AbstractCustomComm> custom_all_reduce_comm_;
    int                                 enable_custom_all_reduce_;

    // function pointer callback
    using callback_sig                 = void(std::unordered_map<std::string, Tensor>*, void*);
    callback_sig* token_generated_cb_  = nullptr;
    void*         token_generated_ctx_ = nullptr;
    

public:
    T* decoder_normed_input_ = nullptr;
    T* self_attn_output_     = nullptr;
    T* ffn_output_           = nullptr;
    T* decoder_layer_output_ = nullptr;

    BaseAttentionLayer<T>* self_attention_layer_;
    FfnLayer<T>*           ffn_layer_;

    // gptj part
    T*       padded_embedding_kernel_;
    T*       padded_embedding_bias_;
    const T* padded_embedding_kernel_ptr_;
    const T* padded_embedding_bias_ptr_;

    T* input_attention_mask_;

    T* decoder_input_buf_;
    T* decoder_output_buf_;
    T* normed_decoder_output_buf_;

    float* logits_buf_;
    float* nccl_logits_buf_;
    float* cum_log_probs_;

    bool* finished_buf_;
    bool* h_finished_buf_;
    int*  sequence_lengths_ = nullptr;

    T*   key_cache_;
    T*   value_cache_;
    int* cache_indirections_[2] = {nullptr, nullptr};

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
    bool*     masked_tokens_             = nullptr;
    uint32_t* seq_limit_len_             = nullptr;
    int*      tiled_total_padding_count_ = nullptr;

    bool* generation_should_stop_ = nullptr;

    T*     context_decoder_input_buf_;
    T*     context_decoder_output_buf_;
    float* output_log_probs_buf_;

    GptJInference(size_t                              max_batch_size,
                size_t                              head_num,
                size_t                              size_per_head,
                size_t                              inter_size,
                size_t                              num_layer,
                int                                 vocab_size,
                size_t                              rotary_embedding_dim,
                bool                                neox_rotary_style,
                float                               layernorm_eps,
                NcclParam                           tensor_para,
                NcclParam                           pipeline_para,
                cudaStream_t                        stream,
                cublasMMWrapper*                    cublas_wrapper,
                bool                                is_free_buffer_after_forward,
                std::shared_ptr<AbstractCustomComm> custom_all_reduce_comm    = nullptr,
                int                                 enable_custom_all_reduce_ = 0);

    ~GptJInference();

    void run(int* stop_flag, const INIReader& reader, const GptJWeight<T>* gpt_weights, InferenceQueue* inque);
             
    void forward(std::unordered_map<std::string, Tensor>*       output_tensors,
                 const std::unordered_map<std::string, Tensor>* input_tensors,
                 const std::vector<GptJDecoderLayerWeight<T>>*  decoder_layer_weights);

};

}  // namespace flover
